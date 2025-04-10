"""
Advanced Reward Functions
======================

This module provides advanced reward functions for quantum circuit optimization,
including rewards that allow for temporary performance decreases to achieve better overall optimization.
"""

import numpy as np
from qiskit import Aer, execute

class AdvancedReward:
    """
    Advanced reward functions for quantum circuit optimization.
    
    This class implements more sophisticated reward mechanisms that can handle
    temporary performance decreases during optimization.
    """
    
    def __init__(self, 
                 depth_weight=0.3,
                 cost_weight=0.3,
                 error_weight=0.4,
                 equivalence_bonus=1.0,
                 lookahead_factor=0.5,
                 history_length=5):
        """
        Initialize the advanced reward calculator.
        
        Args:
            depth_weight (float): Weight for the depth component of the reward.
            cost_weight (float): Weight for the cost component of the reward.
            error_weight (float): Weight for the error component of the reward.
            equivalence_bonus (float): Bonus reward for maintaining circuit equivalence.
            lookahead_factor (float): Factor for weighing potential future improvements.
            history_length (int): Number of past states to consider for trend analysis.
        """
        self.depth_weight = depth_weight
        self.cost_weight = cost_weight
        self.error_weight = error_weight
        self.equivalence_bonus = equivalence_bonus
        self.lookahead_factor = lookahead_factor
        self.history_length = history_length
        
        # History of states for trend analysis
        self.state_history = []
        
    def calculate_immediate_reward(self, state, original_state):
        """
        Calculate the immediate reward for a state compared to the original state.
        
        Args:
            state (CircuitState): The current state.
            original_state (CircuitState): The original state to compare against.
            
        Returns:
            float: The immediate reward.
        """
        # Calculate depth component
        depth_original = original_state.depth
        depth_current = state.depth
        
        if depth_original > 0:
            depth_reward = self.depth_weight * (depth_original / max(1, depth_current) - 1.0)
        else:
            depth_reward = 0.0
            
        # Calculate cost component
        cost_original = original_state.cost
        cost_current = state.cost
        
        if cost_original is not None and cost_current is not None and cost_original > 0:
            cost_reward = self.cost_weight * (cost_original / max(1, cost_current) - 1.0)
        else:
            cost_reward = 0.0
            
        # Calculate error component
        error_original = original_state.error
        error_current = state.error
        
        if error_original is not None and error_current is not None:
            # Lower error is better
            if error_original < 1.0:
                error_reward = self.error_weight * ((1.0 - error_current) / max(0.01, 1.0 - error_original) - 1.0)
            else:
                error_reward = 0.0
        else:
            error_reward = 0.0
            
        # Calculate equivalence bonus
        if state.is_equivalent_to(original_state):
            equivalence_bonus = self.equivalence_bonus
        else:
            equivalence_bonus = 0.0
            
        # Total immediate reward
        total_reward = depth_reward + cost_reward + error_reward + equivalence_bonus
        
        return total_reward
    
    def calculate_trend_reward(self, state):
        """
        Calculate a reward component based on the trend of improvements.
        
        This rewards consistent improvements over time, even if individual steps
        might temporarily decrease performance.
        
        Args:
            state (CircuitState): The current state.
            
        Returns:
            float: The trend reward component.
        """
        # Add current state to history
        self.state_history.append(state)
        
        # Keep only the most recent states
        if len(self.state_history) > self.history_length:
            self.state_history = self.state_history[-self.history_length:]
            
        # If we don't have enough history, return 0
        if len(self.state_history) < 2:
            return 0.0
            
        # Calculate trends for depth, cost, and error
        depth_trend = self._calculate_trend([s.depth for s in self.state_history])
        
        # Cost trend (if available)
        if all(s.cost is not None for s in self.state_history):
            cost_trend = self._calculate_trend([s.cost for s in self.state_history])
        else:
            cost_trend = 0.0
            
        # Error trend (if available)
        if all(s.error is not None for s in self.state_history):
            # For error, lower is better, so negate the trend
            error_trend = -self._calculate_trend([s.error for s in self.state_history])
        else:
            error_trend = 0.0
            
        # Combine trends with weights
        trend_reward = (
            self.depth_weight * depth_trend +
            self.cost_weight * cost_trend +
            self.error_weight * error_trend
        )
        
        return trend_reward
    
    def _calculate_trend(self, values):
        """
        Calculate the trend of a series of values.
        
        A negative trend (decreasing values) is good for depth, cost, and error.
        
        Args:
            values (list): List of values to analyze.
            
        Returns:
            float: The trend value. Negative means decreasing (good).
        """
        if len(values) < 2:
            return 0.0
            
        # Simple linear regression to get the slope
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope using least squares
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        
        numerator = np.sum((x - mean_x) * (y - mean_y))
        denominator = np.sum((x - mean_x) ** 2)
        
        if denominator == 0:
            return 0.0
            
        slope = numerator / denominator
        
        # Normalize the slope by the mean value
        if mean_y != 0:
            normalized_slope = slope / abs(mean_y)
        else:
            normalized_slope = slope
            
        # Return the negative of the normalized slope
        # (negative slope means decreasing values, which is good)
        return -normalized_slope
    
    def calculate_potential_reward(self, state, next_states):
        """
        Calculate a reward component based on potential future improvements.
        
        This allows for temporary performance decreases if they open up
        opportunities for larger improvements later.
        
        Args:
            state (CircuitState): The current state.
            next_states (list): List of potential next states.
            
        Returns:
            float: The potential reward component.
        """
        if not next_states:
            return 0.0
            
        # Calculate immediate rewards for all next states
        next_rewards = [self.calculate_immediate_reward(next_state, state) for next_state in next_states]
        
        # Get the maximum potential reward
        max_potential = max(next_rewards)
        
        # Scale by the lookahead factor
        potential_reward = self.lookahead_factor * max_potential
        
        return potential_reward
    
    def calculate_reward(self, state, original_state, next_states=None):
        """
        Calculate the total reward for a state.
        
        This combines immediate reward, trend reward, and potential reward.
        
        Args:
            state (CircuitState): The current state.
            original_state (CircuitState): The original state to compare against.
            next_states (list, optional): List of potential next states.
            
        Returns:
            float: The total reward.
        """
        # Calculate immediate reward
        immediate_reward = self.calculate_immediate_reward(state, original_state)
        
        # Calculate trend reward
        trend_reward = self.calculate_trend_reward(state)
        
        # Calculate potential reward (if next_states provided)
        if next_states:
            potential_reward = self.calculate_potential_reward(state, next_states)
        else:
            potential_reward = 0.0
            
        # Total reward
        total_reward = immediate_reward + trend_reward + potential_reward
        
        return total_reward
    
    def reset(self):
        """Reset the reward calculator."""
        self.state_history = []

class EquivalenceChecker:
    """
    Advanced equivalence checking for quantum circuits.
    
    This class provides methods to check if two quantum circuits are functionally equivalent,
    using various techniques including simulation, matrix representation, and unitary comparison.
    """
    
    def __init__(self, method='simulation', tolerance=1e-10):
        self.method = method
        self.tolerance = tolerance

    def __repr__(self):
        """Return a string representation of the EquivalenceChecker."""
        return f"EquivalenceChecker(method='{self.method}', tolerance={self.tolerance})"

    @staticmethod
    def check_equivalence_by_simulation(circuit1, circuit2, num_shots=1024):
        """
        Check if two circuits are equivalent by simulating them.
        
        Args:
            circuit1 (QuantumCircuit): First circuit.
            circuit2 (QuantumCircuit): Second circuit.
            num_shots (int): Number of shots for the simulation.
            
        Returns:
            bool: True if the circuits are equivalent, False otherwise.
        """
        # Create a simulator
        simulator = Aer.get_backend('qasm_simulator')
        
        # Execute both circuits
        result1 = execute(circuit1, simulator, shots=num_shots).result()
        result2 = execute(circuit2, simulator, shots=num_shots).result()
        
        # Get the counts
        counts1 = result1.get_counts(circuit1)
        counts2 = result2.get_counts(circuit2)
        
        # Compare the counts
        return counts1 == counts2
    
    @staticmethod
    def check_equivalence_by_unitary(circuit1, circuit2, tolerance=1e-10):
        """
        Check if two circuits are equivalent by comparing their unitary matrices.
        
        Args:
            circuit1 (QuantumCircuit): First circuit.
            circuit2 (QuantumCircuit): Second circuit.
            tolerance (float): Tolerance for numerical comparison.
            
        Returns:
            bool: True if the circuits are equivalent, False otherwise.
        """
        # Create a simulator
        simulator = Aer.get_backend('unitary_simulator')
        
        # Execute both circuits
        result1 = execute(circuit1, simulator).result()
        result2 = execute(circuit2, simulator).result()
        
        # Get the unitary matrices
        unitary1 = result1.get_unitary(circuit1)
        unitary2 = result2.get_unitary(circuit2)
        
        # Compare the unitary matrices
        return np.allclose(unitary1, unitary2, atol=tolerance)
    
    @staticmethod
    def check_equivalence_with_phase(circuit1, circuit2, tolerance=1e-10):
        """
        Check if two circuits are equivalent up to a global phase.
        
        Args:
            circuit1 (QuantumCircuit): First circuit.
            circuit2 (QuantumCircuit): Second circuit.
            tolerance (float): Tolerance for numerical comparison.
            
        Returns:
            bool: True if the circuits are equivalent up to a global phase, False otherwise.
        """
        # Create a simulator
        simulator = Aer.get_backend('unitary_simulator')
        
        # Execute both circuits
        result1 = execute(circuit1, simulator).result()
        result2 = execute(circuit2, simulator).result()
        
        # Get the unitary matrices
        unitary1 = result1.get_unitary(circuit1)
        unitary2 = result2.get_unitary(circuit2)
        
        # Calculate the ratio of the first non-zero elements
        idx = np.nonzero(unitary1)
        if len(idx[0]) == 0:
            return np.allclose(unitary2, 0, atol=tolerance)
            
        i, j = idx[0][0], idx[1][0]
        if abs(unitary2[i, j]) < tolerance:
            return False
            
        phase = unitary1[i, j] / unitary2[i, j]
        
        # Check if unitary2 * phase is close to unitary1
        return np.allclose(unitary1, unitary2 * phase, atol=tolerance)
