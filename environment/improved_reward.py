"""
Improved Reward Functions for Quantum Circuit Optimization
=========================================================

This module provides enhanced reward functions specifically designed for quantum circuit optimization,
with stronger emphasis on gate cancellation, depth reduction, and maintaining circuit equivalence.
"""

import numpy as np
from qiskit import Aer, execute
from .advanced_reward import AdvancedReward

class ImprovedReward(AdvancedReward):
    """
    Improved reward functions for quantum circuit optimization.
    
    This class extends the AdvancedReward class with additional reward components
    specifically tailored for quantum circuit optimization tasks.
    """
    
    def __init__(self, 
                 depth_weight=0.4,
                 cost_weight=0.3,
                 error_weight=0.2,
                 equivalence_bonus=1.5,
                 gate_reduction_weight=0.5,
                 consecutive_improvement_bonus=0.2,
                 exploration_bonus=0.1,
                 lookahead_factor=0.7,
                 history_length=10):
        """
        Initialize the improved reward calculator.
        
        Args:
            depth_weight (float): Weight for the depth component of the reward.
            cost_weight (float): Weight for the cost component of the reward.
            error_weight (float): Weight for the error component of the reward.
            equivalence_bonus (float): Bonus reward for maintaining circuit equivalence.
            gate_reduction_weight (float): Weight for the gate reduction component.
            consecutive_improvement_bonus (float): Bonus for consecutive improvements.
            exploration_bonus (float): Bonus for exploring new circuit configurations.
            lookahead_factor (float): Factor for weighing potential future improvements.
            history_length (int): Number of past states to consider for trend analysis.
        """
        super().__init__(
            depth_weight=depth_weight,
            cost_weight=cost_weight,
            error_weight=error_weight,
            equivalence_bonus=equivalence_bonus,
            lookahead_factor=lookahead_factor,
            history_length=history_length
        )
        
        self.gate_reduction_weight = gate_reduction_weight
        self.consecutive_improvement_bonus = consecutive_improvement_bonus
        self.exploration_bonus = exploration_bonus
        
        # Track consecutive improvements
        self.consecutive_improvements = 0
        
        # Track explored circuit configurations
        self.explored_configurations = set()
        
    def calculate_immediate_reward(self, state, original_state):
        """
        Calculate the immediate reward for a state compared to the original state.
        
        Args:
            state (CircuitState): The current state.
            original_state (CircuitState): The original state to compare against.
            
        Returns:
            float: The immediate reward.
        """
        # Get the basic reward from the parent class
        basic_reward = super().calculate_immediate_reward(state, original_state)
        
        # Calculate gate reduction component
        gate_count_original = len(original_state.circuit_dag.nodes())
        gate_count_current = len(state.circuit_dag.nodes())
        
        if gate_count_original > 0:
            gate_reduction_reward = self.gate_reduction_weight * (
                gate_count_original / max(1, gate_count_current) - 1.0
            )
        else:
            gate_reduction_reward = 0.0
        
        # Calculate consecutive improvement bonus
        if self.state_history and state.depth < self.state_history[-1].depth:
            self.consecutive_improvements += 1
            consecutive_bonus = self.consecutive_improvement_bonus * min(self.consecutive_improvements, 5)
        else:
            self.consecutive_improvements = 0
            consecutive_bonus = 0.0
        
        # Calculate exploration bonus
        circuit_hash = hash(str(state.qiskit_circuit))
        if circuit_hash not in self.explored_configurations:
            self.explored_configurations.add(circuit_hash)
            exploration_bonus = self.exploration_bonus
        else:
            exploration_bonus = 0.0
        
        # Total immediate reward
        total_reward = basic_reward + gate_reduction_reward + consecutive_bonus + exploration_bonus
        
        return total_reward
    
    def calculate_gate_pattern_reward(self, state):
        """
        Calculate reward based on recognizing beneficial gate patterns.
        
        This rewards the agent for creating patterns that are known to be
        optimizable, such as adjacent H-H gates that cancel out.
        
        Args:
            state (CircuitState): The current state.
            
        Returns:
            float: The gate pattern reward.
        """
        # This is a simplified implementation - in practice, you would
        # implement a more sophisticated pattern recognition system
        
        circuit = state.qiskit_circuit
        pattern_reward = 0.0
        
        # Check for adjacent identical single-qubit gates that might cancel
        # This is a very simplified check and would need to be more sophisticated
        # in a real implementation
        for qubit in range(circuit.num_qubits):
            prev_gate = None
            for instruction in circuit.data:
                if len(instruction[1]) == 1 and instruction[1][0].index == qubit:
                    current_gate = instruction[0].name
                    if current_gate == prev_gate and current_gate in ['h', 'x', 'y', 'z']:
                        # Potential cancellation opportunity
                        pattern_reward += 0.1
                    prev_gate = current_gate
        
        return pattern_reward
    
    def calculate_reward(self, state, original_state, next_states=None):
        """
        Calculate the total reward for a state.
        
        This combines immediate reward, trend reward, potential reward,
        and gate pattern reward.
        
        Args:
            state (CircuitState): The current state.
            original_state (CircuitState): The original state to compare against.
            next_states (list, optional): List of potential next states.
            
        Returns:
            float: The total reward.
        """
        # Calculate basic rewards from parent class
        basic_reward = super().calculate_reward(state, original_state, next_states)
        
        # Calculate gate pattern reward
        pattern_reward = self.calculate_gate_pattern_reward(state)
        
        # Total reward
        total_reward = basic_reward + pattern_reward
        
        return total_reward
    
    def reset(self):
        """Reset the reward calculator."""
        super().reset()
        self.consecutive_improvements = 0
        self.explored_configurations = set()
