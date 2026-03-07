"""
Neuro-Symbolic Logic Layer for G-NeSAI

This module implements symbolic reasoning and constraints for explainable
AI in electronic warfare. It provides doctrine rules, action filtering,
and hybrid neuro-symbolic integration.

Key Components:
- SymbolicRuleEngine: Logic-based rule evaluation
- Doctrine constraints: Military rules and ROE (Rules of Engagement)
- Action filtering: Filter neural actions through symbolic constraints
- Explainable decisions: Provide reasoning for actions taken
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from enum import Enum

class ActionType(Enum):
    """Enumeration of possible electronic warfare actions"""
    PASSIVE_MONITOR = 0
    ACTIVE_JAM = 1
    DECEPTIVE_JAM = 2
    DIRECTED_ENERGY = 3
    CYBER_ATTACK = 4
    COORDINATED_ATTACK = 5

class ThreatLevel(Enum):
    """Threat assessment levels"""
    FRIENDLY = 0
    UNKNOWN = 1
    HOSTILE = 2
    HIGH_THREAT = 3

class DoctrineRule:
    """
    Represents a single doctrine rule with conditions and actions.
    """

    def __init__(self, rule_id: str, conditions: Dict, actions: List[ActionType], priority: int = 1):
        """
        Initialize a doctrine rule.

        Args:
            rule_id: Unique identifier for the rule
            conditions: Dictionary of conditions that must be met
            actions: List of allowed actions when conditions are met
            priority: Rule priority (higher = more important)
        """
        self.rule_id = rule_id
        self.conditions = conditions
        self.actions = actions
        self.priority = priority

    def evaluate(self, context: Dict) -> bool:
        """
        Evaluate if the rule conditions are met given the current context.

        Args:
            context: Current situational context

        Returns:
            True if conditions are met, False otherwise
        """
        for condition_key, condition_value in self.conditions.items():
            if condition_key not in context:
                return False

            context_value = context[condition_key]

            # Handle different condition types
            if isinstance(condition_value, dict):
                # Range conditions
                if 'min' in condition_value and context_value < condition_value['min']:
                    return False
                if 'max' in condition_value and context_value > condition_value['max']:
                    return False
                if 'equals' in condition_value and context_value != condition_value['equals']:
                    return False
            else:
                # Direct equality
                if context_value != condition_value:
                    return False

        return True

class SymbolicRuleEngine:
    """
    Engine for evaluating symbolic rules and filtering actions.
    """

    def __init__(self):
        self.rules: List[DoctrineRule] = []
        self._initialize_doctrine_rules()

    def _initialize_doctrine_rules(self):
        """Initialize standard electronic warfare doctrine rules."""

        # Rule 1: High threat - allow aggressive actions
        self.add_rule(DoctrineRule(
            rule_id="high_threat_response",
            conditions={
                'threat_level': ThreatLevel.HIGH_THREAT,
                'own_forces_casualty_risk': {'max': 0.3},
                'mission_criticality': {'min': 0.7}
            },
            actions=[ActionType.ACTIVE_JAM, ActionType.DECEPTIVE_JAM, ActionType.DIRECTED_ENERGY],
            priority=10
        ))

        # Rule 2: Unknown signals - passive monitoring only
        self.add_rule(DoctrineRule(
            rule_id="unknown_signal_monitor",
            conditions={
                'threat_level': ThreatLevel.UNKNOWN,
                'signal_confidence': {'max': 0.6}
            },
            actions=[ActionType.PASSIVE_MONITOR],
            priority=8
        ))

        # Rule 3: Friendly forces protection
        self.add_rule(DoctrineRule(
            rule_id="friendly_protection",
            conditions={
                'friendly_forces_present': True,
                'jamming_radius': {'max': 5000}  # meters
            },
            actions=[ActionType.PASSIVE_MONITOR, ActionType.DECEPTIVE_JAM],
            priority=9
        ))

        # Rule 4: Cyber operations authorization
        self.add_rule(DoctrineRule(
            rule_id="cyber_ops_auth",
            conditions={
                'cyber_authorization': True,
                'target_type': 'command_and_control',
                'collateral_damage_risk': {'max': 0.1}
            },
            actions=[ActionType.CYBER_ATTACK],
            priority=7
        ))

        # Rule 5: Coordinated operations
        self.add_rule(DoctrineRule(
            rule_id="coordinated_ops",
            conditions={
                'allied_forces_coordination': True,
                'communication_links_secure': True
            },
            actions=[ActionType.COORDINATED_ATTACK],
            priority=6
        ))

    def add_rule(self, rule: DoctrineRule):
        """Add a new doctrine rule to the engine."""
        self.rules.append(rule)
        # Sort by priority (highest first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def evaluate_rules(self, context: Dict) -> List[DoctrineRule]:
        """
        Evaluate all rules against the current context.

        Args:
            context: Current situational context

        Returns:
            List of rules that are satisfied
        """
        satisfied_rules = []
        for rule in self.rules:
            if rule.evaluate(context):
                satisfied_rules.append(rule)

        return satisfied_rules

    def get_allowed_actions(self, context: Dict) -> List[ActionType]:
        """
        Get all actions allowed by the satisfied rules.

        Args:
            context: Current situational context

        Returns:
            List of allowed actions
        """
        satisfied_rules = self.evaluate_rules(context)
        allowed_actions = set()

        for rule in satisfied_rules:
            allowed_actions.update(rule.actions)

        return list(allowed_actions)

    def filter_neural_actions(self, neural_actions: torch.Tensor, context: Dict) -> torch.Tensor:
        """
        Filter neural network action probabilities through symbolic constraints.

        Args:
            neural_actions: Action probabilities from neural network
            context: Current situational context

        Returns:
            Filtered action probabilities
        """
        allowed_actions = self.get_allowed_actions(context)

        # Create mask for allowed actions
        mask = torch.zeros_like(neural_actions)

        for action in allowed_actions:
            if action.value < len(mask):
                mask[action.value] = 1.0

        # Apply mask to neural actions
        filtered_actions = neural_actions * mask

        # Renormalize probabilities
        if filtered_actions.sum() > 0:
            filtered_actions = filtered_actions / filtered_actions.sum()

        return filtered_actions

    def explain_decision(self, selected_action: ActionType, context: Dict) -> str:
        """
        Provide explanation for why an action was selected.

        Args:
            selected_action: The action that was chosen
            context: Context in which the decision was made

        Returns:
            Explanation string
        """
        satisfied_rules = self.evaluate_rules(context)
        relevant_rules = [rule for rule in satisfied_rules if selected_action in rule.actions]

        if not relevant_rules:
            return f"Action {selected_action.name} was not authorized by any doctrine rules."

        explanation = f"Action {selected_action.name} was authorized by the following doctrine rules:\n"
        for rule in relevant_rules:
            explanation += f"- {rule.rule_id}: {self._rule_description(rule)}\n"

        return explanation

    def _rule_description(self, rule: DoctrineRule) -> str:
        """Generate human-readable description of a rule."""
        conditions_desc = []
        for key, value in rule.conditions.items():
            if isinstance(value, dict):
                if 'min' in value:
                    conditions_desc.append(f"{key} >= {value['min']}")
                if 'max' in value:
                    conditions_desc.append(f"{key} <= {value['max']}")
                if 'equals' in value:
                    conditions_desc.append(f"{key} == {value['equals']}")
            else:
                conditions_desc.append(f"{key} == {value}")

        actions_desc = [action.name for action in rule.actions]

        return f"When {' and '.join(conditions_desc)}, allow: {', '.join(actions_desc)}"

class NeuroSymbolicIntegrator:
    """
    Integrates neural and symbolic components for hybrid decision-making.
    """

    def __init__(self, neural_model, symbolic_engine: SymbolicRuleEngine):
        """
        Initialize the neuro-symbolic integrator.

        Args:
            neural_model: Neural network model (e.g., Active Inference agent)
            symbolic_engine: Symbolic rule engine
        """
        self.neural_model = neural_model
        self.symbolic_engine = symbolic_engine

    def hybrid_decision(self, observation: torch.Tensor, context: Dict) -> Tuple[ActionType, Dict]:
        """
        Make a hybrid neuro-symbolic decision.

        Args:
            observation: Neural observation input
            context: Symbolic context information

        Returns:
            Tuple of (selected_action, decision_info)
        """
        # Get neural action probabilities
        neural_actions = self.neural_model.get_action_probabilities(observation)

        # Filter through symbolic constraints
        filtered_actions = self.symbolic_engine.filter_neural_actions(neural_actions, context)

        # Select action (argmax of filtered probabilities)
        if filtered_actions.sum() == 0:
            # No actions allowed, default to passive monitoring
            selected_action = ActionType.PASSIVE_MONITOR
            confidence = 0.0
        else:
            action_idx = torch.argmax(filtered_actions).item()
            selected_action = ActionType(action_idx)
            confidence = filtered_actions[action_idx].item()

        # Generate explanation
        explanation = self.symbolic_engine.explain_decision(selected_action, context)

        decision_info = {
            'neural_probabilities': neural_actions.tolist(),
            'filtered_probabilities': filtered_actions.tolist(),
            'confidence': confidence,
            'explanation': explanation,
            'satisfied_rules': len(self.symbolic_engine.evaluate_rules(context))
        }

        return selected_action, decision_info

def create_sample_context() -> Dict:
    """Create a sample context for testing."""
    return {
        'threat_level': ThreatLevel.HIGH_THREAT,
        'own_forces_casualty_risk': 0.2,
        'mission_criticality': 0.8,
        'friendly_forces_present': False,
        'signal_confidence': 0.9,
        'jamming_radius': 3000,
        'cyber_authorization': True,
        'target_type': 'radar',
        'collateral_damage_risk': 0.05,
        'allied_forces_coordination': True,
        'communication_links_secure': True
    }

if __name__ == "__main__":
    # Test the symbolic rule engine
    engine = SymbolicRuleEngine()
    context = create_sample_context()

    print("Testing Symbolic Rule Engine:")
    print(f"Context: {context}")
    print()

    allowed_actions = engine.get_allowed_actions(context)
    print(f"Allowed actions: {[action.name for action in allowed_actions]}")
    print()

    # Test explanation
    explanation = engine.explain_decision(ActionType.ACTIVE_JAM, context)
    print("Explanation for ACTIVE_JAM:")
    print(explanation)