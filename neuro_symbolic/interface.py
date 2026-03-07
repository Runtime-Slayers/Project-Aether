"""
Neuro-symbolic training interface.

Provides a small helper to produce an allowed-class boolean mask
for integration with training loops. It adapts the existing
`cognitive_core.symbolic_rules.SymbolicRuleEngine`.
"""
from typing import List
import torch

from cognitive_core.symbolic_rules import SymbolicRuleEngine, ActionType


class NeuroSymbolicInterface:
    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.engine = SymbolicRuleEngine()

    def get_allowed_mask(self, context: dict) -> torch.Tensor:
        """Return a boolean mask of length `n_classes` where allowed classes are True."""
        allowed = self.engine.get_allowed_actions(context)
        mask = torch.zeros(self.n_classes, dtype=torch.bool)
        for a in allowed:
            if isinstance(a, ActionType):
                idx = a.value
            else:
                try:
                    idx = int(a)
                except Exception:
                    continue
            if 0 <= idx < self.n_classes:
                mask[idx] = True
        return mask
