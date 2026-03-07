import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional
from .pymdp_wrapper import PymdpAdapter

class ActiveInferenceAgent:
    """Lightweight Active Inference component used as a regularizer.

    This implementation provides a simple differentiable free energy estimate
    computed from predicted class probabilities and prior preferences.
    It's intentionally small so it can run on a laptop for demo and publication
    figures; extend with `pymdp` for full A/B/C/D matrix learning.
    """
    def __init__(self, n_classes, prior_pref=None, device='cpu'):
        self.n_classes = n_classes
        if prior_pref is None:
            prior_pref = np.ones(n_classes) / n_classes
        self.prior = torch.tensor(prior_pref, dtype=torch.float32, device=device)
        self.device = device
        # Optional pymdp adapter (lightweight surrogate if pymdp missing)
        try:
            self.adapter = PymdpAdapter(n_states=n_classes, device=device)
        except Exception:
            self.adapter = None

    def free_energy(self, predicted_prob: torch.Tensor, target: torch.Tensor, observation: Optional[torch.Tensor] = None):
        """Compute a simple variational free energy surrogate.

        predicted_prob: [B, C] soft predictions
        target: [B] integer labels
        returns: scalar free energy (torch.Tensor)
        """
        # Negative log likelihood
        eps = 1e-9
        nll = -torch.log(predicted_prob.clamp(eps, 1.0)).gather(1, target.unsqueeze(1)).mean()
        # KL divergence between predicted marginal and prior preferences
        q = predicted_prob.mean(dim=0)
        kl = (q * (torch.log(q.clamp(eps,1.0)) - torch.log(self.prior.clamp(eps,1.0)))).sum()
        fa = nll + kl

        # If adapter available and observation provided, include adapter alignment term
        if getattr(self, 'adapter', None) is not None and observation is not None:
            try:
                adapter_pred = self.adapter.predict(observation)
                adapter_q = adapter_pred.mean(dim=0)
                adapter_kl = (q * (torch.log(q.clamp(eps,1.0)) - torch.log(adapter_q.clamp(eps,1.0)))).sum()
                # small weight to encourage agreement with adapter
                fa = fa + 0.1 * adapter_kl
            except Exception:
                pass

        # Update adapter with posterior beliefs (non-blocking)
        try:
            if getattr(self, 'adapter', None) is not None:
                self.adapter.update(predicted_prob.detach(), observation=observation)
        except Exception:
            pass

        return fa

    def preferred_action_filter(self, predictions: torch.Tensor, allowed_mask: torch.Tensor):
        """Mask out actions not allowed by symbolic doctrine.

        predictions: [B, C] probabilities
        allowed_mask: [C] boolean mask of allowed classes
        returns masked predictions renormalized
        """
        mask = allowed_mask.to(predictions.device).float()
        masked = predictions * mask.unsqueeze(0)
        s = masked.sum(dim=1, keepdim=True).clamp(min=1e-9)
        return masked / s
