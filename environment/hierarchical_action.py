"""
Hierarchical Circuit Action
=========================

This module provides a hierarchical action representation for quantum circuit optimization,
implementing the two-part decomposition approach from the Quarl paper.
"""

import numpy as np
from enum import Enum, auto
from environment.action import ActionType, CircuitAction

class ActionCategory(Enum):
    """High-level categories of circuit transformations."""
    LOCAL_TRANSFORMATION = auto()    # Local gate-level transformations
    GLOBAL_TRANSFORMATION = auto()   # Global circuit-level transformations

class HierarchicalCircuitAction:
    """
    Represents a hierarchical action that can be applied to a quantum circuit during optimization.
    This implements the two-part decomposition approach from the Quarl paper.
    """
    
    def __init__(self, category, action_type, **params):
        """
        Initialize a hierarchical circuit action.
        
        Args:
            category (ActionCategory): The high-level category of the action.
            action_type (ActionType): The specific type of action to perform.
            **params: Additional parameters for the action.
        """
        self.category = category
        self.action_type = action_type
        self.params = params
        
    @classmethod
    def from_indices(cls, category_index, action_index, num_actions, num_qubits, available_gates):
        """
        Create a hierarchical action from category and action indices.
        
        Args:
            category_index (int): The index of the action category.
            action_index (int): The index of the specific action within the category.
            num_actions (int): The total number of possible actions.
            num_qubits (int): The number of qubits in the circuit.
            available_gates (list): List of available gates.
            
        Returns:
            HierarchicalCircuitAction: The hierarchical action.
        """
        # Determine the action category
        if category_index == 0:
            category = ActionCategory.LOCAL_TRANSFORMATION
        else:
            category = ActionCategory.GLOBAL_TRANSFORMATION
        
        # Create the base action using the existing CircuitAction class
        base_action = CircuitAction.from_index(action_index, num_actions, num_qubits, available_gates)
        
        # Create the hierarchical action
        return cls(category, base_action.action_type, **base_action.params)
    
    @classmethod
    def get_num_categories(cls):
        """Get the number of action categories."""
        return len(ActionCategory)
    
    @classmethod
    def get_num_actions_per_category(cls, category, num_qubits, available_gates):
        """
        Get the number of actions for a specific category.
        
        Args:
            category (ActionCategory): The action category.
            num_qubits (int): The number of qubits in the circuit.
            available_gates (list): List of available gates.
            
        Returns:
            int: The number of actions for the category.
        """
        if category == ActionCategory.LOCAL_TRANSFORMATION:
            # Local transformations: gate insertions, removals, and replacements
            num_insert_actions = len(available_gates) * (num_qubits + num_qubits * (num_qubits - 1) // 2)
            num_remove_actions = 100  # Assume max 100 gates
            num_replace_actions = 100 * len(available_gates)
            return num_insert_actions + num_remove_actions + num_replace_actions
        else:
            # Global transformations: decompose, optimize_1q, cancel_2q, consolidate, commute
            return 5 + 99  # 5 basic actions + 99 commute actions
    
    def apply(self, state):
        """
        Apply the hierarchical action to a circuit state.
        
        Args:
            state (CircuitState): The circuit state to apply the action to.
            
        Returns:
            CircuitState: A new circuit state with the action applied.
        """
        # Create a base CircuitAction and apply it
        base_action = CircuitAction(self.action_type, **self.params)
        return base_action.apply(state)
    
    def __str__(self):
        """String representation of the hierarchical action."""
        return f"HierarchicalAction(category={self.category}, type={self.action_type}, params={self.params})"
