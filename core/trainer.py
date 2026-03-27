from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

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
    train_mode: str = "input"          # "input", "recurrent", "all"
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: Optional[float] = 1.0
    device: str = "cpu"
    freeze_trial_inputs: bool = True


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

        self.rnn.set_train_mode(self.cfg.train_mode)

        params = [p for p in self.rnn.parameters() if p.requires_grad]
        if len(params) == 0:
            raise ValueError("No trainable parameters found. Check train_mode.")

        self.optimizer = torch.optim.Adam(
            params,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        self.history = TrainHistory()

        self.fixed_epoch_inputs = None
        self.fixed_epoch_ids = None

        if self.cfg.freeze_trial_inputs:
            self._initialize_fixed_trial_inputs()

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

        x = trial["x"].to(self.cfg.device)

        out = self.rnn(x, noise=rnn_noise)
        states = out["states"]                     # [batch, time, n_rec]
        cursor = self.decoder(states)             # [batch, time]

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
            "epoch_ids": epoch_ids,
            "states": states,
            "cursor": cursor,
            "target": target,
            "weights": weights,
            "loss": loss,
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

        return {
            "loss_value": float(loss.item()),
            "cursor_mean": float(cursor.mean().item()),
            "cursor_std": float(cursor.std().item()),
            "baseline_cursor_mean": _masked_mean(cursor, baseline_mask),
            "task_cursor_mean": _masked_mean(cursor, task_mask),
            "late_cursor_mean": _masked_mean(cursor, late_mask),
            "baseline_target_mean": _masked_mean(target, baseline_mask),
            "task_target_mean": _masked_mean(target, task_mask),
            "late_target_mean": _masked_mean(target, late_mask),
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
            "x": batch["x"][b].detach().cpu(),             # [time, n_inp]
            "states": batch["states"][b].detach().cpu(),   # [time, n_rec]
            "cursor": batch["cursor"][b].detach().cpu(),   # [time]
            "target": batch["target"][b].detach().cpu(),   # [time]
            "epoch_ids": batch["epoch_ids"].detach().cpu(),# [time]
            "axis": self.decoder.axis.detach().cpu().clone(),
            "bci_indices": self.decoder.neuron_indices.detach().cpu().clone(),
            "loss": float(batch["loss"].item()),
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
        rnn_noise: bool = False,
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
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.rnn.parameters() if p.requires_grad],
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
                "baseline_cursor_mean": train_out["baseline_cursor_mean"],
                "task_cursor_mean": train_out["task_cursor_mean"],
                "late_cursor_mean": train_out["late_cursor_mean"],
                "baseline_target_mean": train_out["baseline_target_mean"],
                "task_target_mean": train_out["task_target_mean"],
                "late_target_mean": train_out["late_target_mean"],
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
                    "baseline_cursor_mean": eval_out["baseline_cursor_mean"],
                    "task_cursor_mean": eval_out["task_cursor_mean"],
                    "late_cursor_mean": eval_out["late_cursor_mean"],
                    "baseline_target_mean": eval_out["baseline_target_mean"],
                    "task_target_mean": eval_out["task_target_mean"],
                    "late_target_mean": eval_out["late_target_mean"],
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