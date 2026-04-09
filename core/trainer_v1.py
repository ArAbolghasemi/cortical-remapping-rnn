from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .rnn import MotorCortexRNN
from .decoder import PopulationBCIDecoder
from .task import TrialInputConfig, generate_trial_inputs
from .losses import (
    CursorTargetConfig,
    make_cursor_target,
    weighted_cursor_mse,
)


@dataclass
class TrainerConfig:
    train_mode: str = "all"          # "input", "recurrent", "all", "input_values", "none"
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: Optional[float] = 1.0
    device: str = "cpu"
    freeze_trial_inputs: bool = True

    # input scaling
    input_scaling_mode: str = "cursor_mean"   # "none", "cursor_mean", "cursor_prev"
    input_scale_constant: float = 0.0
    input_scale_gain: float = 1.0
    input_scale_offset: float = 0.0
    detach_scaling_signal: bool = True

    # learned input-drive modification
    input_learning_mode: str = "none"         # "none", "epoch_additive", "epoch_replace"
    input_value_init: str = "zero"            # "zero", "fixed"
    input_value_scale: float = 1.0            # scale for zero init if needed


@dataclass
class TrainHistory:
    records: List[Dict[str, Any]] = field(default_factory=list)
    snapshots: List[Dict[str, Any]] = field(default_factory=list)

    def append(self, record: Dict[str, Any]) -> None:
        self.records.append(record)

    def append_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self.snapshots.append(snapshot)

    def last(self) -> Dict[str, Any]:
        if len(self.records) == 0:
            raise ValueError("History is empty.")
        return self.records[-1]

    def to_dict(self) -> Dict[str, List[Any]]:
        if len(self.records) == 0:
            return {}
        keys = self.records[0].keys()
        return {k: [r[k] for r in self.records] for k in keys}


class BCITrainer:
    def __init__(
        self,
        rnn: MotorCortexRNN,
        decoder: PopulationBCIDecoder,
        trial_cfg: TrialInputConfig,
        target_cfg: CursorTargetConfig,
        trainer_cfg: TrainerConfig,
    ):
        self.rnn = rnn
        self.decoder = decoder
        self.trial_cfg = trial_cfg
        self.target_cfg = target_cfg
        self.cfg = trainer_cfg

        self.rnn.to(self.cfg.device)
        self.decoder.to(self.cfg.device)

        self.history = TrainHistory()

        self.fixed_epoch_inputs = None
        self.fixed_epoch_ids = None

        if self.cfg.freeze_trial_inputs:
            self._initialize_fixed_trial_inputs()

        self.learned_epoch_inputs = nn.ParameterDict()
        self._initialize_learned_input_values()

        self.rnn.set_train_mode(self._rnn_train_mode_from_cfg(self.cfg.train_mode))
        self._set_input_value_requires_grad(self.cfg.train_mode)

        self.optimizer = self._build_optimizer()

    def _rnn_train_mode_from_cfg(self, mode: str) -> str:
        mode = mode.lower()
        if mode == "input_values":
            return "none"
        return mode

    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self.rnn.parameters() if p.requires_grad]
        params += [p for p in self.learned_epoch_inputs.parameters() if p.requires_grad]

        if len(params) == 0:
            raise ValueError("No trainable parameters found. Check train_mode / input_learning_mode.")

        return torch.optim.Adam(
            params,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

    def _set_input_value_requires_grad(self, mode: str) -> None:
        mode = mode.lower()
        requires_grad = (mode == "input_values") and (self.cfg.input_learning_mode != "none")

        for p in self.learned_epoch_inputs.parameters():
            p.requires_grad_(requires_grad)

    def set_train_mode(self, mode: str) -> None:
        self.cfg.train_mode = mode
        self.rnn.set_train_mode(self._rnn_train_mode_from_cfg(mode))
        self._set_input_value_requires_grad(mode)
        self.optimizer = self._build_optimizer()

    def _initialize_fixed_trial_inputs(self) -> None:
        trial = generate_trial_inputs(
            batch_size=1,
            cfg=self.trial_cfg,
            noise=False,
        )
        self.fixed_epoch_inputs = {
            "baseline": trial["epoch_inputs"]["baseline"].detach().clone().to(self.cfg.device),
            "task": trial["epoch_inputs"]["task"].detach().clone().to(self.cfg.device),
            "late": trial["epoch_inputs"]["late"].detach().clone().to(self.cfg.device),
        }
        self.fixed_epoch_ids = trial["epoch_ids"].detach().clone().to(self.cfg.device)

    def _initialize_learned_input_values(self) -> None:
        n_inp = self.trial_cfg.n_inp

        def _make_param_from_init(epoch_name: str) -> nn.Parameter:
            if self.cfg.input_value_init == "fixed":
                if self.fixed_epoch_inputs is None:
                    raise ValueError(
                        "input_value_init='fixed' requires freeze_trial_inputs=True so fixed inputs exist."
                    )
                src = self.fixed_epoch_inputs[epoch_name].reshape(-1).detach().clone()
                if src.numel() != n_inp:
                    raise ValueError(
                        f"Expected fixed epoch input '{epoch_name}' to flatten to shape [{n_inp}], "
                        f"got {tuple(self.fixed_epoch_inputs[epoch_name].shape)}"
                    )
                val = src.to(self.cfg.device)
            elif self.cfg.input_value_init == "zero":
                val = torch.zeros(n_inp, device=self.cfg.device) * self.cfg.input_value_scale
            else:
                raise ValueError(f"Unknown input_value_init: {self.cfg.input_value_init}")

            return nn.Parameter(val)

        self.learned_epoch_inputs = nn.ParameterDict({
            "baseline": _make_param_from_init("baseline"),
            "task": _make_param_from_init("task"),
            "late": _make_param_from_init("late"),
        })

    @torch.no_grad()
    def reset_learned_input_values(self) -> None:
        self._initialize_learned_input_values()
        self._set_input_value_requires_grad(self.cfg.train_mode)
        self.optimizer = self._build_optimizer()

    @torch.no_grad()
    def set_learned_epoch_input(self, epoch_name: str, value: torch.Tensor) -> None:
        if epoch_name not in self.learned_epoch_inputs:
            raise ValueError(f"Unknown epoch_name: {epoch_name}")

        value = value.to(self.cfg.device).reshape(-1)
        n_inp = self.trial_cfg.n_inp
        if value.numel() != n_inp:
            raise ValueError(f"value must have {n_inp} elements, got {value.numel()}")

        self.learned_epoch_inputs[epoch_name].data.copy_(value)

    def _apply_learned_input_modification(
        self,
        x_raw: torch.Tensor,
        epoch_ids: torch.Tensor,
    ) -> torch.Tensor:
        mode = self.cfg.input_learning_mode

        if mode == "none":
            return x_raw

        x_mod = x_raw.clone()

        epoch_param_map = {
            0: self.learned_epoch_inputs["baseline"],
            1: self.learned_epoch_inputs["task"],
            2: self.learned_epoch_inputs["late"],
        }

        for epoch_idx, param in epoch_param_map.items():
            mask = (epoch_ids == epoch_idx)
            if mask.sum() == 0:
                continue

            if mode == "epoch_additive":
                x_mod[:, mask, :] = x_mod[:, mask, :] + param.view(1, 1, -1)
            elif mode == "epoch_replace":
                x_mod[:, mask, :] = param.view(1, 1, -1)
            else:
                raise ValueError(f"Unknown input_learning_mode: {mode}")

        return x_mod

    def _compute_input_scale(
        self,
        cursor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mode = self.cfg.input_scaling_mode

        if mode == "none":
            return torch.tensor(1.0, device=self.cfg.device)

        if mode == "cursor_mean":
            if cursor is None:
                raise ValueError("cursor must be provided when input_scaling_mode='cursor_mean'.")

            cursor_mean = cursor.mean()
            if self.cfg.detach_scaling_signal:
                cursor_mean = cursor_mean.detach()

            scale = (
                self.cfg.input_scale_offset
                + self.cfg.input_scale_constant
                + self.cfg.input_scale_gain * cursor_mean
            )
            scale = torch.clamp(scale, min=0.1, max=10.0)
            return scale

        raise ValueError(f"Unknown input_scaling_mode: {mode}")

    @torch.no_grad()
    def resample_fixed_trial_inputs(self) -> None:
        self._initialize_fixed_trial_inputs()

    @torch.no_grad()
    def set_decoder_axis(self, new_axis: torch.Tensor) -> None:
        self.decoder.set_axis(new_axis.to(self.cfg.device))

    @torch.no_grad()
    def resample_decoder_axis(self) -> None:
        self.decoder.resample_axis()

    def _forward_batch(
        self,
        batch_size: int,
        input_noise: bool,
        rnn_noise: bool,
    ) -> Dict[str, torch.Tensor]:

        if self.cfg.freeze_trial_inputs:
            trial = generate_trial_inputs(
                batch_size=batch_size,
                cfg=self.trial_cfg,
                noise=input_noise,
                baseline_input=self.fixed_epoch_inputs["baseline"],
                task_input=self.fixed_epoch_inputs["task"],
                late_input=self.fixed_epoch_inputs["late"],
            )
            epoch_ids = self.fixed_epoch_ids
        else:
            trial = generate_trial_inputs(
                batch_size=batch_size,
                cfg=self.trial_cfg,
                noise=input_noise,
            )
            epoch_ids = trial["epoch_ids"].to(self.cfg.device)

        x_raw = trial["x"].to(self.cfg.device)   # [B, T, n_in]
        x_pre_scale = self._apply_learned_input_modification(x_raw, epoch_ids)

        B, T, _ = x_raw.shape

        if self.cfg.input_scaling_mode == "none":
            input_scale = torch.ones(B, T, device=self.cfg.device)
            x = x_pre_scale

            out = self.rnn(x, noise=rnn_noise)
            states = out["states"]
            cursor = self.decoder(states)

        elif self.cfg.input_scaling_mode == "cursor_mean":
            out_pre = self.rnn(x_pre_scale, noise=rnn_noise)
            states_pre = out_pre["states"]
            cursor_pre = self.decoder(states_pre)

            scalar_scale = self._compute_input_scale(cursor_pre)
            x = x_pre_scale * scalar_scale
            input_scale = scalar_scale.expand(B, T)

            out = self.rnn(x, noise=rnn_noise)
            states = out["states"]
            cursor = self.decoder(states)

        elif self.cfg.input_scaling_mode == "cursor_prev":
            h = self.rnn.initial_state(batch_size=B)

            states_list = []
            cursor_list = []
            x_scaled_list = []
            scale_list = []

            prev_cursor = torch.zeros(B, device=self.cfg.device, dtype=x_raw.dtype)

            for t in range(T):
                if self.cfg.detach_scaling_signal:
                    prev_cursor_for_scale = prev_cursor.detach()
                else:
                    prev_cursor_for_scale = prev_cursor

                scale_t = (
                    self.cfg.input_scale_offset
                    + self.cfg.input_scale_constant
                    + self.cfg.input_scale_gain * prev_cursor_for_scale
                )
                scale_t = torch.clamp(scale_t, min=0.1, max=10.0)

                x_t = x_pre_scale[:, t, :] * scale_t[:, None]
                h = self.rnn.step(h, x_t, noise=rnn_noise)
                cursor_t = self.decoder(h.unsqueeze(1)).squeeze(1)

                states_list.append(h)
                cursor_list.append(cursor_t)
                x_scaled_list.append(x_t)
                scale_list.append(scale_t)

                prev_cursor = cursor_t

            states = torch.stack(states_list, dim=1)
            cursor = torch.stack(cursor_list, dim=1)
            x = torch.stack(x_scaled_list, dim=1)
            input_scale = torch.stack(scale_list, dim=1)

        else:
            raise ValueError(f"Unknown input_scaling_mode: {self.cfg.input_scaling_mode}")

        target_out = make_cursor_target(
            batch_size=batch_size,
            epoch_ids=epoch_ids,
            cfg=self.target_cfg,
        )
        target = target_out["target"].to(self.cfg.device)
        weights = target_out["weights"].to(self.cfg.device)

        loss = weighted_cursor_mse(cursor, target, weights)

        return {
            "x": x,
            "x_raw": x_raw,
            "x_pre_scale": x_pre_scale,
            "input_scale": input_scale,
            "epoch_ids": epoch_ids,
            "states": states,
            "cursor": cursor,
            "target": target,
            "weights": weights,
            "loss": loss,
            "learned_input_baseline": self.learned_epoch_inputs["baseline"],
            "learned_input_task": self.learned_epoch_inputs["task"],
            "learned_input_late": self.learned_epoch_inputs["late"],
        }

    def _compute_metrics(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        cursor = batch["cursor"]
        target = batch["target"]
        epoch_ids = batch["epoch_ids"]
        loss = batch["loss"]

        baseline_mask = epoch_ids == 0
        task_mask = epoch_ids == 1
        late_mask = epoch_ids == 2

        def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> float:
            return float(x[:, mask].mean().item())

        input_scale = batch.get("input_scale", torch.ones_like(cursor))

        return {
            "loss_value": float(loss.item()),
            "cursor_mean": float(cursor.mean().item()),
            "cursor_std": float(cursor.std().item()),
            "input_scale_mean": float(input_scale.mean().item()),
            "input_scale_std": float(input_scale.std().item()),
            "baseline_cursor_mean": _masked_mean(cursor, baseline_mask),
            "task_cursor_mean": _masked_mean(cursor, task_mask),
            "late_cursor_mean": _masked_mean(cursor, late_mask),
            "baseline_target_mean": _masked_mean(target, baseline_mask),
            "task_target_mean": _masked_mean(target, task_mask),
            "late_target_mean": _masked_mean(target, late_mask),
            "learned_input_baseline_norm": float(batch["learned_input_baseline"].norm().item()),
            "learned_input_task_norm": float(batch["learned_input_task"].norm().item()),
            "learned_input_late_norm": float(batch["learned_input_late"].norm().item()),
        }

    def _make_snapshot(
        self,
        batch: Dict[str, torch.Tensor],
        step: int,
        phase_name: str,
        mode: str,
        save_batch_index: int = 0,
    ) -> Dict[str, Any]:
        b = save_batch_index
        return {
            "step": step,
            "phase": phase_name,
            "mode": mode,
            "x": batch["x"][b].detach().cpu(),
            "x_raw": batch["x_raw"][b].detach().cpu() if "x_raw" in batch else None,
            "x_pre_scale": batch["x_pre_scale"][b].detach().cpu() if "x_pre_scale" in batch else None,
            "states": batch["states"][b].detach().cpu(),
            "cursor": batch["cursor"][b].detach().cpu(),
            "target": batch["target"][b].detach().cpu(),
            "epoch_ids": batch["epoch_ids"].detach().cpu(),
            "axis": self.decoder.axis.detach().cpu().clone(),
            "bci_indices": self.decoder.neuron_indices.detach().cpu().clone(),
            "loss": float(batch["loss"].item()),
            "input_scale": batch["input_scale"][b].detach().cpu() if "input_scale" in batch else None,
            "learned_input_baseline": self.learned_epoch_inputs["baseline"].detach().cpu().clone(),
            "learned_input_task": self.learned_epoch_inputs["task"].detach().cpu().clone(),
            "learned_input_late": self.learned_epoch_inputs["late"].detach().cpu().clone(),
        }

    @torch.no_grad()
    def probe(
        self,
        batch_size: int = 1,
        input_noise: bool = False,
        rnn_noise: bool = False,
        step: int = -1,
        phase_name: str = "probe",
    ) -> Dict[str, Any]:
        self.rnn.eval()

        batch = self._forward_batch(
            batch_size=batch_size,
            input_noise=input_noise,
            rnn_noise=rnn_noise,
        )
        metrics = self._compute_metrics(batch)

        out = {
            **batch,
            **metrics,
            "mode": "probe",
            "phase": phase_name,
            "step": step,
        }
        return out

    @torch.no_grad()
    def evaluate(
        self,
        batch_size: int = 32,
        input_noise: bool = False,
        rnn_noise: bool = False,
        step: int = -1,
        phase_name: str = "eval",
        save_snapshot: bool = False,
    ) -> Dict[str, Any]:
        self.rnn.eval()

        batch = self._forward_batch(
            batch_size=batch_size,
            input_noise=input_noise,
            rnn_noise=rnn_noise,
        )
        metrics = self._compute_metrics(batch)

        out = {
            **batch,
            **metrics,
            "mode": "eval",
            "phase": phase_name,
            "step": step,
        }

        if save_snapshot:
            self.history.append_snapshot(
                self._make_snapshot(batch, step=step, phase_name=phase_name, mode="eval")
            )

        return out

    def train_step(
        self,
        batch_size: int = 32,
        input_noise: bool = True,
        rnn_noise: bool = True,
        step: int = -1,
        phase_name: str = "train",
        save_snapshot: bool = False,
    ) -> Dict[str, Any]:
        self.rnn.train()
        self.optimizer.zero_grad()

        batch = self._forward_batch(
            batch_size=batch_size,
            input_noise=input_noise,
            rnn_noise=rnn_noise,
        )

        loss = batch["loss"]
        loss.backward()

        grad_norm = None
        if self.cfg.grad_clip_norm is not None:
            trainable_params = [p for p in self.rnn.parameters() if p.requires_grad]
            trainable_params += [p for p in self.learned_epoch_inputs.parameters() if p.requires_grad]

            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=self.cfg.grad_clip_norm,
            )
            grad_norm = float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm)

        self.optimizer.step()

        metrics = self._compute_metrics(batch)
        out = {
            **batch,
            **metrics,
            "grad_norm": grad_norm,
            "mode": "train",
            "phase": phase_name,
            "step": step,
        }

        if save_snapshot:
            self.history.append_snapshot(
                self._make_snapshot(batch, step=step, phase_name=phase_name, mode="train")
            )

        return out

    def fit_phase(
        self,
        n_steps: int,
        phase_name: str,
        start_step: int = 0,
        batch_size: int = 32,
        eval_every: Optional[int] = 10,
        eval_batch_size: int = 64,
        input_noise_train: bool = True,
        rnn_noise_train: bool = True,
        input_noise_eval: bool = False,
        rnn_noise_eval: bool = False,
        print_every: Optional[int] = 10,
        save_train_snapshot_every: Optional[int] = None,
        save_eval_snapshots: bool = True,
    ) -> TrainHistory:
        for local_step in range(1, n_steps + 1):
            global_step = start_step + local_step

            save_train_snapshot = (
                save_train_snapshot_every is not None and
                (local_step % save_train_snapshot_every == 0 or local_step == 1 or local_step == n_steps)
            )

            train_out = self.train_step(
                batch_size=batch_size,
                input_noise=input_noise_train,
                rnn_noise=rnn_noise_train,
                step=global_step,
                phase_name=phase_name,
                save_snapshot=save_train_snapshot,
            )

            record = {
                "step": global_step,
                "phase": phase_name,
                "mode": "train",
                "loss": train_out["loss_value"],
                "cursor_mean": train_out["cursor_mean"],
                "cursor_std": train_out["cursor_std"],
                "input_scale_mean": train_out["input_scale_mean"],
                "input_scale_std": train_out["input_scale_std"],
                "baseline_cursor_mean": train_out["baseline_cursor_mean"],
                "task_cursor_mean": train_out["task_cursor_mean"],
                "late_cursor_mean": train_out["late_cursor_mean"],
                "baseline_target_mean": train_out["baseline_target_mean"],
                "task_target_mean": train_out["task_target_mean"],
                "late_target_mean": train_out["late_target_mean"],
                "learned_input_baseline_norm": train_out["learned_input_baseline_norm"],
                "learned_input_task_norm": train_out["learned_input_task_norm"],
                "learned_input_late_norm": train_out["learned_input_late_norm"],
                "grad_norm": train_out["grad_norm"],
            }
            self.history.append(record)

            if print_every is not None and (local_step % print_every == 0 or local_step == 1 or local_step == n_steps):
                print(
                    f"[{phase_name} | train step {global_step:04d}] "
                    f"loss={record['loss']:.6f} | "
                    f"base={record['baseline_cursor_mean']:.4f}, "
                    f"task={record['task_cursor_mean']:.4f}, "
                    f"late={record['late_cursor_mean']:.4f}"
                )

            if eval_every is not None and local_step % eval_every == 0:
                eval_out = self.evaluate(
                    batch_size=eval_batch_size,
                    input_noise=input_noise_eval,
                    rnn_noise=rnn_noise_eval,
                    step=global_step,
                    phase_name=phase_name,
                    save_snapshot=save_eval_snapshots,
                )
                eval_record = {
                    "step": global_step,
                    "phase": phase_name,
                    "mode": "eval",
                    "loss": eval_out["loss_value"],
                    "cursor_mean": eval_out["cursor_mean"],
                    "cursor_std": eval_out["cursor_std"],
                    "input_scale_mean": eval_out["input_scale_mean"],
                    "input_scale_std": eval_out["input_scale_std"],
                    "baseline_cursor_mean": eval_out["baseline_cursor_mean"],
                    "task_cursor_mean": eval_out["task_cursor_mean"],
                    "late_cursor_mean": eval_out["late_cursor_mean"],
                    "baseline_target_mean": eval_out["baseline_target_mean"],
                    "task_target_mean": eval_out["task_target_mean"],
                    "late_target_mean": eval_out["late_target_mean"],
                    "learned_input_baseline_norm": eval_out["learned_input_baseline_norm"],
                    "learned_input_task_norm": eval_out["learned_input_task_norm"],
                    "learned_input_late_norm": eval_out["learned_input_late_norm"],
                    "grad_norm": None,
                }
                self.history.append(eval_record)

                if print_every is not None:
                    print(
                        f"[{phase_name} | eval  step {global_step:04d}] "
                        f"loss={eval_record['loss']:.6f} | "
                        f"base={eval_record['baseline_cursor_mean']:.4f}, "
                        f"task={eval_record['task_cursor_mean']:.4f}, "
                        f"late={eval_record['late_cursor_mean']:.4f}"
                    )

        return self.history