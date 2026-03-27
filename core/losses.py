from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class CursorTargetConfig:
    # baseline and late targets are always constant
    baseline_target: float = 0.0
    late_target: float = 1.0

    # task target can be shaped over time
    task_target_mode: str = "linear"   # "flat" or "linear"
    task_target_value: float = 0.5     # only used if mode == "flat"

    # epoch weights in the loss
    baseline_weight: float = 0.5
    task_weight: float = 1.0
    late_weight: float = 5.0

    device: str = "cpu"
    dtype: torch.dtype = torch.float32


def _make_task_target(
    task_len: int,
    cfg: CursorTargetConfig,
) -> torch.Tensor:
    """
    Create the target trajectory for the task epoch.

    Modes:
        - "flat"   : constant task_target_value
        - "linear" : ramp from baseline_target to late_target
    """
    if task_len <= 0:
        raise ValueError(f"task_len must be positive, got {task_len}")

    mode = cfg.task_target_mode.lower()

    if mode == "flat":
        task_target = torch.full(
            (task_len,),
            fill_value=cfg.task_target_value,
            device=cfg.device,
            dtype=cfg.dtype,
        )
        return task_target

    elif mode == "linear":
        if task_len == 1:
            return torch.tensor(
                [cfg.late_target],
                device=cfg.device,
                dtype=cfg.dtype,
            )

        task_target = torch.linspace(
            cfg.baseline_target,
            cfg.late_target,
            steps=task_len,
            device=cfg.device,
            dtype=cfg.dtype,
        )
        return task_target

    else:
        raise ValueError(
            f"Unsupported task_target_mode: {cfg.task_target_mode}. "
            f"Supported modes are: 'flat', 'linear'."
        )


def make_cursor_target(
    batch_size: int,
    epoch_ids: torch.Tensor,
    cfg: CursorTargetConfig,
) -> Dict[str, torch.Tensor]:
    """
    Build target cursor trajectory and per-time-step loss weights.

    Args:
        batch_size: number of trials in batch
        epoch_ids : [time] with
                    0 = baseline
                    1 = task
                    2 = late

    Returns:
        dict with:
            target  : [batch, time]
            weights : [batch, time]
    """
    if epoch_ids.ndim != 1:
        raise ValueError(f"epoch_ids must have shape [time], got {tuple(epoch_ids.shape)}")

    T = epoch_ids.shape[0]

    baseline_mask = epoch_ids == 0
    task_mask = epoch_ids == 1
    late_mask = epoch_ids == 2

    n_baseline = int(baseline_mask.sum().item())
    n_task = int(task_mask.sum().item())
    n_late = int(late_mask.sum().item())

    if n_baseline == 0 or n_task == 0 or n_late == 0:
        raise ValueError(
            "epoch_ids must contain all three epochs: 0 (baseline), 1 (task), 2 (late)"
        )

    target_1d = torch.empty(T, device=cfg.device, dtype=cfg.dtype)
    weights_1d = torch.empty(T, device=cfg.device, dtype=cfg.dtype)

    # Baseline target: constant
    target_1d[baseline_mask] = cfg.baseline_target

    # Task target: configurable shape
    task_target = _make_task_target(n_task, cfg)
    target_1d[task_mask] = task_target

    # Late target: constant
    target_1d[late_mask] = cfg.late_target

    # Loss weights by epoch
    weights_1d[baseline_mask] = cfg.baseline_weight
    weights_1d[task_mask] = cfg.task_weight
    weights_1d[late_mask] = cfg.late_weight

    target = target_1d.unsqueeze(0).repeat(batch_size, 1)
    weights = weights_1d.unsqueeze(0).repeat(batch_size, 1)

    return {
        "target": target,
        "weights": weights,
    }


def weighted_cursor_mse(
    cursor: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Weighted MSE between cursor and target.

    Args:
        cursor : [batch, time]
        target : [batch, time]
        weights: [batch, time] or None

    Returns:
        scalar loss
    """
    if cursor.shape != target.shape:
        raise ValueError(
            f"cursor and target must have same shape, got {tuple(cursor.shape)} vs {tuple(target.shape)}"
        )

    sq_err = (cursor - target) ** 2

    if weights is None:
        return sq_err.mean()

    if weights.shape != cursor.shape:
        raise ValueError(
            f"weights must match cursor shape, got {tuple(weights.shape)} vs {tuple(cursor.shape)}"
        )

    return (weights * sq_err).sum() / weights.sum().clamp_min(1e-12)