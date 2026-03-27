import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RNNConfig:
    n_inp: int = 10
    n_rec: int = 200
    dt: float = 0.01
    tau: float = 0.1
    noise_std: float = 0.01
    w_inp_scale: float = 0.1
    w_rec_scale: float = 0.5
    nonlinearity: str = "tanh"
    device: str = "cpu"
    dtype: torch.dtype = torch.float32


class MotorCortexRNN(nn.Module):
    """
    Vanilla continuous-time firing-rate RNN with Euler discretization:

        h_{t+1} = h_t + (dt/tau) * [ -h_t + phi(W_rec h_t + W_inp x_t) + noise ]

    Shapes:
        x        : [batch, time, n_inp]
        h0       : [batch, n_rec]
        h_t      : [batch, n_rec]
        states   : [batch, time, n_rec]
    """

    def __init__(self, cfg: RNNConfig):
        super().__init__()
        self.cfg = cfg
        self.alpha = cfg.dt / cfg.tau

        self.n_inp = cfg.n_inp
        self.n_rec = cfg.n_rec
        self.noise_std = cfg.noise_std

        # Parameters
        self.W_inp = nn.Parameter(
            torch.empty(cfg.n_rec, cfg.n_inp, device=cfg.device, dtype=cfg.dtype)
        )
        self.W_rec = nn.Parameter(
            torch.empty(cfg.n_rec, cfg.n_rec, device=cfg.device, dtype=cfg.dtype)
        )
        self.b_rec = nn.Parameter(
            torch.zeros(cfg.n_rec, device=cfg.device, dtype=cfg.dtype)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize:
            W_rec ~ (w_rec_scale / sqrt(n_rec)) * N(0,1)
            W_inp ~ (w_inp_scale / sqrt(n_inp)) * N(0,1)
            b_rec = 0
        """
        with torch.no_grad():
            self.W_rec.normal_(mean=0.0, std=self.cfg.w_rec_scale / math.sqrt(self.n_rec))
            self.W_inp.normal_(mean=0.0, std=self.cfg.w_inp_scale / math.sqrt(self.n_inp))
            self.b_rec.zero_()

    def phi(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.nonlinearity == "tanh":
            return torch.tanh(x)
        elif self.cfg.nonlinearity == "relu":
            return F.relu(x)
        else:
            raise ValueError(f"Unsupported nonlinearity: {self.cfg.nonlinearity}")

    def initial_state(
        self,
        batch_size: int,
        h0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if h0 is not None:
            if h0.shape != (batch_size, self.n_rec):
                raise ValueError(
                    f"h0 must have shape {(batch_size, self.n_rec)}, got {tuple(h0.shape)}"
                )
            return h0
        return torch.zeros(
            batch_size,
            self.n_rec,
            device=self.W_rec.device,
            dtype=self.W_rec.dtype,
        )

    def step(
        self,
        h: torch.Tensor,
        x_t: torch.Tensor,
        noise: bool = True,
    ) -> torch.Tensor:
        """
        Single Euler update.

        Args:
            h   : [batch, n_rec]
            x_t : [batch, n_inp]
        Returns:
            h_next : [batch, n_rec]
        """
        if h.ndim != 2 or x_t.ndim != 2:
            raise ValueError("h and x_t must both be rank-2 tensors")

        rec_term = h @ self.W_rec.T                  # [batch, n_rec]
        inp_term = x_t @ self.W_inp.T               # [batch, n_rec]
        preact = rec_term + inp_term + self.b_rec
        rate = self.phi(preact)

        if noise and self.noise_std > 0:
            eta = self.noise_std * torch.randn_like(h)
        else:
            eta = torch.zeros_like(h)

        h_next = h + self.alpha * (-h + rate + eta)
        return h_next

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        noise: bool = True,
        return_preact: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Run the RNN over a full trial/block.

        Args:
            x : [batch, time, n_inp]
        Returns:
            dict with:
                "states" : [batch, time, n_rec]
                "h_final": [batch, n_rec]
                optionally "preacts": [batch, time, n_rec]
        """
        if x.ndim != 3:
            raise ValueError(f"x must have shape [batch, time, n_inp], got {tuple(x.shape)}")
        if x.shape[-1] != self.n_inp:
            raise ValueError(f"Last dim of x must be {self.n_inp}, got {x.shape[-1]}")

        batch_size, T, _ = x.shape
        h = self.initial_state(batch_size, h0=h0)

        states = []
        preacts = [] if return_preact else None

        for t in range(T):
            x_t = x[:, t, :]
            rec_term = h @ self.W_rec.T
            inp_term = x_t @ self.W_inp.T
            preact = rec_term + inp_term + self.b_rec
            rate = self.phi(preact)

            if noise and self.noise_std > 0:
                eta = self.noise_std * torch.randn_like(h)
            else:
                eta = torch.zeros_like(h)

            h = h + self.alpha * (-h + rate + eta)

            states.append(h)
            if return_preact:
                preacts.append(preact)

        out = {
            "states": torch.stack(states, dim=1),   # [batch, time, n_rec]
            "h_final": h,                           # [batch, n_rec]
        }

        if return_preact:
            out["preacts"] = torch.stack(preacts, dim=1)

        return out

    def freeze_recurrent(self) -> None:
        self.W_rec.requires_grad_(False)
        self.b_rec.requires_grad_(False)

    def unfreeze_recurrent(self) -> None:
        self.W_rec.requires_grad_(True)
        self.b_rec.requires_grad_(True)

    def freeze_input(self) -> None:
        self.W_inp.requires_grad_(False)

    def unfreeze_input(self) -> None:
        self.W_inp.requires_grad_(True)

    def set_train_mode(self, mode: str) -> None:
        """
        mode:
            "all"       -> train W_inp, W_rec, b_rec
            "input"     -> train W_inp only
            "recurrent" -> train W_rec and b_rec only
            "none"      -> freeze everything
        """
        mode = mode.lower()

        if mode == "all":
            self.unfreeze_input()
            self.unfreeze_recurrent()
        elif mode == "input":
            self.unfreeze_input()
            self.freeze_recurrent()
        elif mode == "recurrent":
            self.freeze_input()
            self.unfreeze_recurrent()
        elif mode == "none":
            self.freeze_input()
            self.freeze_recurrent()
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @torch.no_grad()
    def effective_jacobian_at(
        self,
        h: torch.Tensor,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Approximate local Jacobian d h_{t+1} / d h_t for one batch element.
        Only supports batch size = 1.
        Useful later for checking whether recurrent dynamics really changed.

        Args:
            h   : [1, n_rec]
            x_t : [1, n_inp]
        Returns:
            J : [n_rec, n_rec]
        """
        if h.shape[0] != 1 or x_t.shape[0] != 1:
            raise ValueError("effective_jacobian_at currently expects batch size = 1")

        preact = h @ self.W_rec.T + x_t @ self.W_inp.T + self.b_rec  # [1, n_rec]

        if self.cfg.nonlinearity == "tanh":
            deriv = 1.0 - torch.tanh(preact).pow(2)
        elif self.cfg.nonlinearity == "relu":
            deriv = (preact > 0).to(preact.dtype)
        else:
            raise ValueError(f"Unsupported nonlinearity: {self.cfg.nonlinearity}")

        D = torch.diag(deriv.squeeze(0))  # [n_rec, n_rec]
        I = torch.eye(self.n_rec, device=h.device, dtype=h.dtype)
        J = (1.0 - self.alpha) * I + self.alpha * (D @ self.W_rec)
        return J

