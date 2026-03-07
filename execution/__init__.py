# Project AETHER - Execution Module
# Training, Inference, and Visualization Pipelines

"""
Execution Module: Orchestrates training and deployment of G-NeSAI agents.

Provides:
- Training loops with Free Energy minimization
- Visualization of belief updates and policy inference
- Real-time spectrum environment simulation
- Performance metrics and logging
"""

from .trainer import GNeSAITrainer, train_cognitive_agent
from .visualizer import FreeEnergyVisualizer, plot_belief_landscape

__all__ = [
    'GNeSAITrainer',
    'train_cognitive_agent',
    'FreeEnergyVisualizer', 
    'plot_belief_landscape'
]
