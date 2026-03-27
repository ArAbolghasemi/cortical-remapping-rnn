import torch

from core import (
    RNNConfig,
    MotorCortexRNN,
    BCIDecoderConfig,
    PopulationBCIDecoder,
)

# Build model
rnn_cfg = RNNConfig(n_inp=10, n_rec=200)
model = MotorCortexRNN(rnn_cfg)

# Build decoder
dec_cfg = BCIDecoderConfig(n_rec=200, n_bci=50)
decoder = PopulationBCIDecoder(dec_cfg)

# Fake input
x = torch.randn(4, 100, 10)

# Run RNN
out = model(x, noise=False)
states = out["states"]   # [4, 100, 200]

# Decode cursor
cursor = decoder(states)  # [4, 100]

print("states shape:", states.shape)
print("cursor shape:", cursor.shape)
print("num BCI neurons:", len(decoder.neuron_indices))
print("axis norm:", decoder.axis.norm().item())
print("nonzero axis entries:", (decoder.axis.abs() > 0).sum().item())