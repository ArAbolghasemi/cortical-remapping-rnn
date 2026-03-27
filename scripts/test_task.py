from core import TrialInputConfig, generate_trial_inputs

cfg = TrialInputConfig(n_inp=10)
out = generate_trial_inputs(batch_size=4, cfg=cfg, noise=True)

print("x shape:", out["x"].shape)
print("epoch_ids shape:", out["epoch_ids"].shape)
print("epoch slices:", out["epoch_slices"])
print("epoch inputs:,", out["epoch_inputs"])

from core import CursorTargetConfig, make_cursor_target
import torch

epoch_ids = torch.tensor([0]*30 + [1]*60 + [2]*30)

out = make_cursor_target(
    batch_size=2,
    epoch_ids=epoch_ids,
    cfg=CursorTargetConfig(task_target_mode="linear")
)

print(out["target"].shape)
print(out["weights"].shape)
print(out["target"][0, :35])
print(out["target"][0, 85:95])