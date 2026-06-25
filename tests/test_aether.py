import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import torch

from neuro_symbolic.interface import NeuroSymbolicInterface
from cognitive_core.symbolic_rules import create_sample_context
from cognitive_core.pymdp_wrapper import PymdpAdapter

def test_imports():
    from perception_layer.complex_net import ComplexValuedCNN
    from data_factory.generator import SignalGenerator
    assert ComplexValuedCNN is not None
    assert SignalGenerator is not None

def test_neuro_symbolic():
    ctx = create_sample_context()
    iface = NeuroSymbolicInterface(n_classes=6)
    mask = iface.get_allowed_mask(ctx)
    assert mask.sum() > 0, 'No allowed actions — check rules/context'

def test_pymdp_adapter():
    adapter = PymdpAdapter(n_states=5, device='cpu', lr=1e-2)
    obs = torch.randn(4, 16)
    pred = adapter.predict(obs)
    assert pred.shape == (4, 5)
    
    posterior = torch.softmax(torch.randn(4, 5), dim=-1)
    adapter.update(posterior, observation=obs)
