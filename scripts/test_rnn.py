import torch
from core import RNNConfig, MotorCortexRNN

cfg = RNNConfig()
model = MotorCortexRNN(cfg)

x = torch.randn(4, 100, cfg.n_inp)
out = model(x)

print(out["states"].shape)
print(out["h_final"].shape)