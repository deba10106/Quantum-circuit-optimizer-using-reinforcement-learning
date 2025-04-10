"""
GNN-based Quantum Environment
==========================

This module provides an enhanced reinforcement learning environment for quantum circuit optimization
with GNN-based state representation and hierarchical action space.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from qiskit import QuantumCircuit

from environment.state import CircuitState
from environment.action import CircuitAction, ActionType
from environment.hierarchical_action import HierarchicalCircuitAction, ActionCategory
from cost_database.cost_database import CostDatabase

class GNNQuantumEnvironment(gym.Env):
    """
    A reinforcement learning environment for quantum circuit optimization with GNN-based
    state representation and hierarchical action space.
    
    This environment allows an agent to apply transformations to a quantum circuit
    with the goal of optimizing its depth, cost, and error rate.
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 initial_circuit=None, 
                 cost_database=None, 
                 max_steps=100,
                 depth_weight=0.3,
                 cost_weight=0.3,
                 error_weight=0.4,
                 equivalence_bonus=1.0,
                 node_feature_dim=16,
                 render_mode=None,
                 use_hierarchical_actions=True):
        """
        Initialize the GNN-based quantum environment.
        
        Args:
            initial_circuit: A Qiskit QuantumCircuit or our QuantumCircuit wrapper to optimize.
                If None, a random circuit will be generated.
            cost_database: A CostDatabase instance for calculating costs and errors.
                If None, a default database will be created.
            max_steps (int): Maximum number of steps per episode.
            depth_weight (float): Weight for the depth component of the reward.
            cost_weight (float): Weight for the cost component of the reward.
            error_weight (float): Weight for the error component of the reward.
            equivalence_bonus (float): Bonus reward for maintaining circuit equivalence.
            node_feature_dim (int): Dimension of node features for GNN.
            render_mode (str): The render mode to use.
            use_hierarchical_actions (bool): Whether to use hierarchical actions.
        """
        super().__init__()
        
        # Store parameters
        self.max_steps = max_steps
        self.depth_weight = depth_weight
        self.cost_weight = cost_weight
        self.error_weight = error_weight
        self.equivalence_bonus = equivalence_bonus
        self.node_feature_dim = node_feature_dim
        self.render_mode = render_mode
        self.use_hierarchical_actions = use_hierarchical_actions
        
        # Create a cost database if not provided
        if cost_database is None:
            self.cost_database = CostDatabase()
        else:
            self.cost_database = cost_database
        
        # Define available gates
        self.available_gates = ['x', 'y', 'z', 'h', 'cx', 'cz', 'swap', 'rz', 'rx', 'ry', 'u1', 'u2', 'u3', 'ecr']
        
        # Create initial circuit if not provided
        if initial_circuit is None:
            self.initial_circuit = self._generate_random_circuit(5, 20)
        else:
            self.initial_circuit = initial_circuit
        
        # Create initial state
        self.initial_state = CircuitState(self.initial_circuit, self.cost_database)
        
        # Set current state
        self.state = None
        self.original_state = None
        self.steps = 0
        
        # Get number of qubits
        self.num_qubits = self.initial_state.qiskit_circuit.num_qubits
        
        # Define action space
        num_actions = CircuitAction.get_num_actions(self.num_qubits, self.available_gates)
        self.action_space = spaces.Discrete(num_actions)
        
        # Define hierarchical action space
        if use_hierarchical_actions:
            self.num_categories = HierarchicalCircuitAction.get_num_categories()
            self.num_actions_per_category = [
                HierarchicalCircuitAction.get_num_actions_per_category(
                    category, self.num_qubits, self.available_gates
                )
                for category in list(ActionCategory)
            ]
        
        # Define observation space (for compatibility with Gym)
        # In practice, we'll use the graph representation directly
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(node_feature_dim * 10,), dtype=np.float32
        )
        
        # For tracking best circuit
        self.best_state = None
        self.best_reward = -np.inf
        
    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.
        
        Args:
            seed (int, optional): Random seed.
            options (dict, optional): Additional options.
            
        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset state
        self.state = self.initial_state.copy()
        self.original_state = self.initial_state.copy()
        self.steps = 0
        
        # Reset best circuit
        self.best_state = self.state.copy()
        self.best_reward = self._calculate_reward(self.state, self.original_state)
        
        # Get observation
        observation = self._get_observation()
        
        # Return observation and info
        info = self._get_info()
        return observation, info
    
    def step(self, action):
        """
        Take a step in the environment by applying an action.
        
        Args:
            action: The action to take. Can be an index or a HierarchicalCircuitAction.
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Handle hierarchical actions
        if self.use_hierarchical_actions and isinstance(action, tuple):
            category_index, action_index = action
            action = HierarchicalCircuitAction.from_indices(
                category_index, action_index, self.action_space.n, self.num_qubits, self.available_gates
            )
        elif self.use_hierarchical_actions and isinstance(action, HierarchicalCircuitAction):
            # Action is already a HierarchicalCircuitAction
            pass
        else:
            # Convert action index to CircuitAction
            action = CircuitAction.from_index(action, self.action_space.n, self.num_qubits, self.available_gates)
        
        # Apply action to get new state
        try:
            new_state = action.apply(self.state)
        except Exception as e:
            # If action fails, return current state with negative reward
            print(f"Action failed: {str(e)}")
            observation = self._get_observation()
            reward = -0.1  # Penalty for failed action
            terminated = False
            truncated = self.steps >= self.max_steps
            info = self._get_info()
            return observation, reward, terminated, truncated, info
        
        # Calculate reward
        reward = self._calculate_reward(new_state, self.original_state)
        
        # Update state
        self.state = new_state
        self.steps += 1
        
        # Update best circuit if this one is better
        if reward > self.best_reward:
            self.best_state = self.state.copy()
            self.best_reward = reward
        
        # Check if episode is done
        terminated = False  # We don't terminate based on a goal
        truncated = self.steps >= self.max_steps
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        # Render if needed
        if self.render_mode == 'human':
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        """
        Get the observation representation of the current state.
        
        Returns:
            tuple: (node_features, edge_indices) for GNN-based agents.
        """
        # Get the graph representation
        node_features, edge_indices, _ = self.state.get_graph_representation()
        
        # Ensure node features have the right dimension
        if node_features.shape[1] != self.node_feature_dim:
            # Pad or truncate to the right dimension
            if node_features.shape[1] < self.node_feature_dim:
                padding = np.zeros((node_features.shape[0], self.node_feature_dim - node_features.shape[1]))
                node_features = np.hstack((node_features, padding))
            else:
                node_features = node_features[:, :self.node_feature_dim]
        
        return node_features, edge_indices
    
    def _calculate_reward(self, state, original_state):
        """
        Calculate the reward for a state.
        
        The reward is a weighted combination of:
        - Reduction in circuit depth
        - Reduction in circuit cost
        - Reduction in circuit error
        - Bonus for maintaining circuit equivalence
        
        Args:
            state (CircuitState): The current state.
            original_state (CircuitState): The original state to compare against.
            
        Returns:
            float: The reward.
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
            
        # Total reward
        total_reward = depth_reward + cost_reward + error_reward + equivalence_bonus
        
        return total_reward
    
    def _get_info(self):
        """Get additional information about the current state."""
        return {
            'depth': self.state.depth,
            'gate_count': self.state.gate_count,
            'two_qubit_gate_count': self.state.two_qubit_gate_count,
            'cost': self.state.cost,
            'error': self.state.error,
            'best_reward': self.best_reward
        }
    
    def _generate_random_circuit(self, num_qubits, depth):
        """Generate a random quantum circuit."""
        from qiskit.circuit.random import random_circuit
        return random_circuit(num_qubits, depth, measure=False)
    
    def render(self):
        """Render the current state of the environment."""
        if self.render_mode == 'human':
            print(f"Step: {self.steps}")
            print(f"Depth: {self.state.depth}")
            print(f"Gate count: {self.state.gate_count}")
            print(f"Two-qubit gate count: {self.state.two_qubit_gate_count}")
            print(f"Cost: {self.state.cost}")
            print(f"Error: {self.state.error}")
            print(f"Best reward: {self.best_reward}")
            print(f"Circuit:\n{self.state.qiskit_circuit}")
            print("-" * 80)
    
    def close(self):
        """Clean up resources."""
        pass
