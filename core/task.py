from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class TrialInputConfig:
    n_inp: int = 10

    # epoch lengths in time steps
    t_baseline: int = 30
    t_task: int = 60
    t_late: int = 30

    # scale of random epoch input vectors
    baseline_scale: float = 0.3
    task_scale: float = 1.0
    late_scale: float = 0.6

    # noise added directly to inputs
    input_noise_std: float = 0.05

    device: str = "cpu"
    dtype: torch.dtype = torch.float32


def _make_epoch_vector(
    n_inp: int,
    scale: float,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    v = scale * torch.randn(n_inp, device=device, dtype=dtype)
    return v


def generate_trial_inputs(
    batch_size: int,
    cfg: TrialInputConfig,
    noise: bool = True,
    baseline_input: Optional[torch.Tensor] = None,
    task_input: Optional[torch.Tensor] = None,
    late_input: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Generate piecewise-constant batched trial inputs.

    Each epoch is defined by a full n_inp-dimensional vector.
    If not provided, that vector is randomly sampled.

    Returns:
        x            : [batch, time, n_inp]
        epoch_ids    : [time]
        epoch_slices : dict of slices
        epoch_inputs : dict of the 3 epoch vectors
    """
    T = cfg.t_baseline + cfg.t_task + cfg.t_late
    x = torch.zeros(batch_size, T, cfg.n_inp, device=cfg.device, dtype=cfg.dtype)

    sl_baseline = slice(0, cfg.t_baseline)
    sl_task = slice(cfg.t_baseline, cfg.t_baseline + cfg.t_task)
    sl_late = slice(cfg.t_baseline + cfg.t_task, T)

    if baseline_input is None:
        baseline_input = _make_epoch_vector(
            cfg.n_inp, cfg.baseline_scale, cfg.device, cfg.dtype
        )
    else:
        baseline_input = baseline_input.to(device=cfg.device, dtype=cfg.dtype)

    if task_input is None:
        task_input = _make_epoch_vector(
            cfg.n_inp, cfg.task_scale, cfg.device, cfg.dtype
        )
    else:
        task_input = task_input.to(device=cfg.device, dtype=cfg.dtype)

    if late_input is None:
        late_input = _make_epoch_vector(
            cfg.n_inp, cfg.late_scale, cfg.device, cfg.dtype
        )
    else:
        late_input = late_input.to(device=cfg.device, dtype=cfg.dtype)

    x[:, sl_baseline, :] = baseline_input.view(1, 1, -1)
    x[:, sl_task, :] = task_input.view(1, 1, -1)
    x[:, sl_late, :] = late_input.view(1, 1, -1)

    if noise and cfg.input_noise_std > 0:
        x = x + cfg.input_noise_std * torch.randn_like(x)

    epoch_ids = torch.empty(T, device=cfg.device, dtype=torch.long)
    epoch_ids[sl_baseline] = 0
    epoch_ids[sl_task] = 1
    epoch_ids[sl_late] = 2

    return {
        "x": x,
        "epoch_ids": epoch_ids,
        "epoch_slices": {
            "baseline": sl_baseline,
            "task": sl_task,
            "late": sl_late,
        },
        "epoch_inputs": {
            "baseline": baseline_input,
            "task": task_input,
            "late": late_input,
        },
    }