import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class BCIDecoderConfig:
    n_rec: int = 200
    n_bci: int = 50
    device: str = "cpu"
    dtype: torch.dtype = torch.float32


class PopulationBCIDecoder(nn.Module):
    """
    Population BCI decoder:
        y_t = h_t @ a

    where:
    - h_t is hidden activity (..., n_rec)
    - a is a unit-norm decoder axis of shape (n_rec,)
    - a is nonzero only on a chosen subset of BCI neurons
    """

    def __init__(
        self,
        cfg: BCIDecoderConfig,
        neuron_indices: Optional[torch.Tensor] = None,
        axis: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.n_rec = cfg.n_rec
        self.n_bci = cfg.n_bci

        if neuron_indices is None:
            perm = torch.randperm(self.n_rec, device=cfg.device)
            neuron_indices = perm[: self.n_bci]
        else:
            neuron_indices = neuron_indices.to(device=cfg.device)
            if neuron_indices.ndim != 1 or len(neuron_indices) != self.n_bci:
                raise ValueError(
                    f"neuron_indices must have shape ({self.n_bci},), got {tuple(neuron_indices.shape)}"
                )

        # Save BCI neuron subset as a buffer, not a trainable parameter
        self.register_buffer("neuron_indices", neuron_indices.long())

        mask = torch.zeros(self.n_rec, device=cfg.device, dtype=cfg.dtype)
        mask[self.neuron_indices] = 1.0
        self.register_buffer("mask", mask)

        if axis is None:
            axis = self._make_random_axis()
        else:
            axis = axis.to(device=cfg.device, dtype=cfg.dtype)
            axis = self._sanitize_axis(axis)

        # Keep decoder axis as buffer by default.
        # It is part of task definition, not something we train with gradient descent.
        self.register_buffer("axis", axis)

    def _make_random_axis(self) -> torch.Tensor:
        axis = torch.zeros(self.n_rec, device=self.mask.device, dtype=self.mask.dtype)
        axis_vals = torch.randn(self.n_bci, device=self.mask.device, dtype=self.mask.dtype)
        axis[self.neuron_indices] = axis_vals
        axis = axis / axis.norm(p=2).clamp_min(1e-12)
        return axis

    def _sanitize_axis(self, axis: torch.Tensor) -> torch.Tensor:
        if axis.shape != (self.n_rec,):
            raise ValueError(f"axis must have shape ({self.n_rec},), got {tuple(axis.shape)}")

        # Force support only on BCI neurons
        axis = axis * self.mask

        norm = axis.norm(p=2)
        if norm < 1e-12:
            raise ValueError("Axis norm is zero after masking; invalid decoder axis.")

        axis = axis / norm
        return axis

    @torch.no_grad()
    def set_axis(self, new_axis: torch.Tensor) -> None:
        new_axis = new_axis.to(device=self.axis.device, dtype=self.axis.dtype)
        new_axis = self._sanitize_axis(new_axis)
        self.axis.copy_(new_axis)

    @torch.no_grad()
    def resample_axis(self) -> None:
        self.axis.copy_(self._make_random_axis())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute cursor from hidden states.

        Args:
            hidden_states:
                shape (..., n_rec)
                e.g.
                - [batch, time, n_rec]
                - [time, n_rec]
                - [n_rec]

        Returns:
            cursor:
                shape (...)
                e.g.
                - [batch, time]
                - [time]
                - scalar
        """
        if hidden_states.shape[-1] != self.n_rec:
            raise ValueError(
                f"Last dim of hidden_states must be {self.n_rec}, got {hidden_states.shape[-1]}"
            )

        cursor = torch.matmul(hidden_states, self.axis)
        return cursor

    def extra_repr(self) -> str:
        return f"n_rec={self.n_rec}, n_bci={self.n_bci}"