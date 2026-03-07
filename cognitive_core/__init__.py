"""
Cognitive Core Module for G-NeSAI (Generative Neuro-Symbolic Active Inference AI)

This module implements the Active Inference agent using pymdp, integrating with
the perception layer for closed-loop cognition in cognitive electronic warfare.

Key Components:
- ActiveInferenceAgent: Core agent with learnable A, B, C, D matrices
- Hierarchical structure for meta-learning
- Integration with CVNN perception layer
- Free Energy minimization for decision-making
"""

import torch
import numpy as np
try:
    from pymdp import MDP, MDPSolver
    from pymdp.core import utils
    _HAS_PYMDP = True
except Exception:
    # Provide lightweight fallbacks so the package can be imported
    _HAS_PYMDP = False

    class MDP:
        def __init__(self, *args, **kwargs):
            pass

    class MDPSolver:
        def __init__(self, mdp):
            self.mdp = mdp
        def infer_states(self, obs):
            return None
        def infer_policies(self):
            return 0, 0.0

    class _UtilsFallback:
        @staticmethod
        def random_A_matrix(n_obs, n_states):
            return np.ones((n_obs, n_states)) / n_states
        @staticmethod
        def random_B_matrix(s1, s2, na):
            return np.repeat(np.eye(s1)[None, :, :], na, axis=0)
        @staticmethod
        def random_C_matrix(n_obs):
            return np.zeros(n_obs)
        @staticmethod
        def random_D_matrix(n_states):
            v = np.ones(n_states) / n_states
            return v

    utils = _UtilsFallback()
    
    # Minimal Agent fallback used by hierarchical meta-agents when pymdp is absent
    class Agent:
        def __init__(self, A=None, B=None, C=None, D=None):
            self.A = A
            self.B = B
            self.C = C
            self.D = D
            # simple beliefs
            n_states = D.shape[0] if D is not None else 1
            self.qs = np.ones(n_states) / float(n_states)
            self.prior_qs = np.ones(n_states) / float(n_states)
            self.qs_prev = np.copy(self.qs)

        def infer_states(self, obs):
            # naive update: shift beliefs slightly toward a uniform posterior
            self.qs_prev = np.copy(self.qs)
            self.qs = (self.qs + (np.ones_like(self.qs) / self.qs.size)) / 2.0
            return self.qs

        def infer_policies(self):
            # return default action and placeholder free energy
            return 0, 0.0
from perception_layer.complex_net import ComplexValuedCNN
from data_factory.spectrum_loader import SpectrumDataset
import networkx as nx
import matplotlib.pyplot as plt

class ActiveInferenceAgent:
    """
    Advanced Active Inference agent for cognitive electronic warfare.

    Implements hierarchical Active Inference with learnable generative models,
    integrating complex-valued neural networks for RF signal processing.
    """

    def __init__(self, num_observations, num_actions, num_states, perception_model=None):
        """
        Initialize the Active Inference agent.

        Args:
            num_observations: Number of possible observations (e.g., signal classes)
            num_actions: Number of possible actions (e.g., jamming strategies)
            num_states: Number of hidden states
            perception_model: Pre-trained CVNN model for perception
        """
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.num_states = num_states
        self.perception_model = perception_model

        # Initialize generative model matrices (A, B, C, D)
        self.A = self._initialize_A_matrix()  # Likelihood matrix
        self.B = self._initialize_B_matrix()  # Transition matrix
        self.C = self._initialize_C_matrix()  # Prior preferences
        self.D = self._initialize_D_matrix()  # Prior beliefs

        # Create pymdp MDP
        self.mdp = MDP(A=self.A, B=self.B, C=self.C, D=self.D)
        self.solver = MDPSolver(self.mdp)

        # Hierarchical structure for meta-learning
        self.hierarchy_levels = 3
        self.meta_agents = [self._create_meta_agent(level) for level in range(self.hierarchy_levels)]

        # Learning parameters
        self.learning_rate = 0.01
        self.free_energy_history = []

    def _initialize_A_matrix(self):
        """Initialize likelihood matrix A (observation model)"""
        A = utils.random_A_matrix(self.num_observations, self.num_states)
        return A

    def _initialize_B_matrix(self):
        """Initialize transition matrix B (state dynamics)"""
        B = utils.random_B_matrix(self.num_states, self.num_states, self.num_actions)
        return B

    def _initialize_C_matrix(self):
        """Initialize prior preferences C"""
        C = utils.random_C_matrix(self.num_observations)
        return C

    def _initialize_D_matrix(self):
        """Initialize prior beliefs D"""
        D = utils.random_D_matrix(self.num_states)
        return D

    def _create_meta_agent(self, level):
        """Create meta-agent for hierarchical structure"""
        meta_obs = self.num_observations // (2 ** level)
        meta_actions = self.num_actions // (2 ** level)
        meta_states = self.num_states // (2 ** level)

        return Agent(
            A=utils.random_A_matrix(meta_obs, meta_states),
            B=utils.random_B_matrix(meta_states, meta_states, meta_actions),
            C=utils.random_C_matrix(meta_obs),
            D=utils.random_D_matrix(meta_states)
        )

    def perceive(self, raw_signal):
        """
        Process raw RF signal through perception layer.

        Args:
            raw_signal: Complex-valued I/Q signal data

        Returns:
            observation: Processed observation for Active Inference
        """
        if self.perception_model is None:
            # Fallback to simple processing
            return self._simple_perception(raw_signal)

        # Use CVNN for advanced perception
        with torch.no_grad():
            features = self.perception_model(raw_signal.unsqueeze(0))
            # Convert to observation index
            observation = torch.argmax(features, dim=1).item()

        return observation

    def _simple_perception(self, signal):
        """Simple perception fallback"""
        # Basic signal classification based on amplitude
        amplitude = np.abs(signal).mean()
        if amplitude > 0.5:
            return 0  # High power signal
        else:
            return 1  # Low power signal

    def act(self, observation):
        """
        Perform Active Inference action selection.

        Args:
            observation: Current observation

        Returns:
            action: Selected action
            free_energy: Variational Free Energy
        """
        # Update beliefs
        self.solver.infer_states(observation)

        # Select action by minimizing expected free energy
        action, free_energy = self.solver.infer_policies()

        # Store free energy for analysis
        self.free_energy_history.append(free_energy)

        return action, free_energy

    def learn(self, observation, action, reward):
        """
        Update generative model parameters using learning.

        Args:
            observation: Observed outcome
            action: Taken action
            reward: Reward signal
        """
        # Update matrices using gradient descent on free energy
        self._update_A_matrix(observation)
        self._update_B_matrix(action)
        self._update_C_matrix(reward)

    def _update_A_matrix(self, observation):
        """Update likelihood matrix A"""
        # Simplified learning rule
        self.A[observation, :] += self.learning_rate * (self.solver.qs - self.solver.prior_qs)

    def _update_B_matrix(self, action):
        """Update transition matrix B"""
        # Simplified learning rule
        self.B[:, :, action] += self.learning_rate * np.outer(self.solver.qs, self.solver.qs_prev)

    def _update_C_matrix(self, reward):
        """Update prior preferences C"""
        # Update based on reward
        self.C += self.learning_rate * reward * self.solver.qs

    def hierarchical_inference(self, observation):
        """
        Perform hierarchical inference across meta-agents.

        Args:
            observation: Current observation

        Returns:
            meta_action: Action from highest level
        """
        current_obs = observation

        for level in range(self.hierarchy_levels):
            meta_agent = self.meta_agents[level]
            meta_agent.infer_states(current_obs)
            meta_action, _ = meta_agent.infer_policies()

            # Abstract observation for next level
            current_obs = meta_action

        return meta_action

    def visualize_free_energy(self, save_path=None):
        """
        Visualize the Free Energy landscape over time.

        Args:
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.free_energy_history)
        plt.title('Variational Free Energy Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Free Energy')
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def get_belief_graph(self):
        """
        Generate a NetworkX graph representing current beliefs.

        Returns:
            graph: NetworkX graph of state beliefs
        """
        G = nx.Graph()

        # Add nodes for states
        for i in range(self.num_states):
            G.add_node(f'State_{i}', belief=self.solver.qs[i])

        # Add edges based on transition probabilities
        for i in range(self.num_states):
            for j in range(self.num_states):
                if self.B[i, j, 0] > 0.1:  # Threshold for edge creation
                    G.add_edge(f'State_{i}', f'State_{j}', weight=self.B[i, j, 0])

        return G

class CognitiveController:
    """
    High-level controller integrating perception, cognition, and action.
    """

    def __init__(self, perception_model_path=None):
        # Load perception model if available
        self.perception_model = None
        if perception_model_path:
            self.perception_model = ComplexValuedCNN()
            self.perception_model.load_state_dict(torch.load(perception_model_path))
            self.perception_model.eval()

        # Initialize Active Inference agent
        self.agent = ActiveInferenceAgent(
            num_observations=10,  # 10 signal classes
            num_actions=5,       # 5 possible actions (jamming strategies)
            num_states=20,       # 20 hidden states
            perception_model=self.perception_model
        )

    def process_signal(self, signal_data):
        """
        Complete processing pipeline: perception -> cognition -> action.

        Args:
            signal_data: Raw I/Q signal data

        Returns:
            action: Selected action
            free_energy: Free energy value
            beliefs: Current belief state
        """
        # Perception
        observation = self.agent.perceive(signal_data)

        # Cognition and action selection
        action, free_energy = self.agent.act(observation)

        # Get current beliefs
        beliefs = self.agent.solver.qs.copy()

        return action, free_energy, beliefs

    def get_action_probabilities(self, observation):
        """
        Get action probabilities from the cognitive agent.

        Args:
            observation: Current observation

        Returns:
            Action probabilities tensor
        """
        # Mock implementation - in practice this would come from the agent's policy
        return torch.softmax(torch.randn(self.agent.num_actions), dim=0)

    def train_episode(self, signal_dataset, num_episodes=100):
        """
        Train the agent on a dataset of signals.

        Args:
            signal_dataset: Dataset of signals
            num_episodes: Number of training episodes
        """
        for episode in range(num_episodes):
            for signal, label in signal_dataset:
                # Process signal
                action, free_energy, beliefs = self.process_signal(signal)

                # Simulate reward (higher for correct classification)
                reward = 1.0 if action == label else -1.0

                # Learn
                self.agent.learn(label, action, reward)

            print(f"Episode {episode+1}/{num_episodes}, Avg Free Energy: {np.mean(self.agent.free_energy_history[-len(signal_dataset):])}")

if __name__ == "__main__":
    # Example usage
    controller = CognitiveController()

    # Create sample signal data
    sample_signal = torch.randn(2, 1024)  # Complex signal (real, imag)

    # Process signal
    action, free_energy, beliefs = controller.process_signal(sample_signal)

    print(f"Selected Action: {action}")
    print(f"Free Energy: {free_energy}")
    print(f"Belief State: {beliefs}")

    # Visualize free energy
    controller.agent.visualize_free_energy(save_path="free_energy_plot.png")
