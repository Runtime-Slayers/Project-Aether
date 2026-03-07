"""
Optional pymdp wrapper for Active Inference components.

This module provides a thin adapter that uses `pymdp` when available,
and falls back to a minimal, pure-PyTorch surrogate when not.

API:
- `PymdpAdapter(n_states, n_obs)` exposes `predict` and `update` hooks
  used by `ActiveInferenceAgent`.
"""
from typing import Optional
import numpy as np
import torch
import torch.nn as nn

try:
    import pymdp
    HAS_PYMDP = True
except Exception:
    HAS_PYMDP = False


class PymdpAdapter:
    """Adapter that uses `pymdp` when available and a small PyTorch surrogate otherwise.

    The surrogate is a single linear mapping from flattened observations -> state logits
    with a tiny online update rule so it can adapt during training/online use.
    """

    def __init__(self, n_states: int, n_obs: Optional[int] = None, device: str = 'cpu', lr: float = 1e-2):
        self.n_states = n_states
        self.device = torch.device(device)
        self.use_pymdp = HAS_PYMDP
        self.lr = float(lr)

        # PyMDP placeholders
        if self.use_pymdp:
            try:
                # user can extend to create proper pymdp A/B/C/D
                self._pymdp_model = None
            except Exception:
                self.use_pymdp = False

        # Surrogate model (created lazily once we see an observation shape)
        self.obs_dim = None
        self.surrogate: Optional[nn.Linear] = None

    def _ensure_surrogate(self, x: torch.Tensor):
        if self.surrogate is None:
            D = int(x.view(x.shape[0], -1).shape[1])
            self.obs_dim = D
            self.surrogate = nn.Linear(D, self.n_states).to(self.device)

    def predict(self, observation: torch.Tensor) -> torch.Tensor:
        """Return a distribution over states for a batch of observations.

        observation: torch.Tensor shape [B, ...]
        returns: torch.Tensor shape [B, n_states]
        """
        if self.use_pymdp:
            # A real pymdp integration would go here; keep uniform fallback
            B = observation.shape[0]
            return torch.full((B, self.n_states), 1.0 / self.n_states, device=self.device)

        # Surrogate path
        x = observation.to(self.device).view(observation.shape[0], -1).float()
        self._ensure_surrogate(x)
        logits = self.surrogate(x)
        probs = torch.softmax(logits, dim=-1)
        return probs

    def update(self, posterior: torch.Tensor, observation: Optional[torch.Tensor] = None):
        """Lightweight online update to align the surrogate to a target posterior.

        posterior: torch.Tensor shape [B, n_states]
        observation: optional tensor [B, ...] used to compute update
        """
        if self.use_pymdp:
            # No-op: user may implement pymdp learning updates here
            return

        if observation is None or self.surrogate is None:
            return

        x = observation.to(self.device).view(observation.shape[0], -1).float()
        with torch.no_grad():
            pred = torch.softmax(self.surrogate(x), dim=-1)
            err = posterior.to(self.device) - pred  # [B, S]
            # Compute simple least-squares update for weights: grad_w = err^T @ x / B
            B = x.shape[0]
            grad_w = (err.t() @ x) / float(max(1, B))  # [S, D]
            self.surrogate.weight.add_(self.lr * grad_w)
            # Update bias
            grad_b = err.mean(dim=0)
            self.surrogate.bias.add_(self.lr * grad_b)

