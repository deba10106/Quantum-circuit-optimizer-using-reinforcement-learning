"""
Quantum Environment
=================

This module provides the reinforcement learning environment for quantum circuit optimization.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit

from environment.state import CircuitState
from environment.action import CircuitAction, ActionType
from cost_database.cost_database import CostDatabase

class QuantumEnvironment(gym.Env):
    """
    A reinforcement learning environment for quantum circuit optimization.
    
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
                 feature_dim=32,
                 render_mode=None):
        """
        Initialize the quantum environment.
        
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
            feature_dim (int): Dimension of the state feature vector.
            render_mode (str): The render mode to use.
        """
        super().__init__()
        
        # Store parameters
        self.max_steps = max_steps
        self.depth_weight = depth_weight
        self.cost_weight = cost_weight
        self.error_weight = error_weight
        self.equivalence_bonus = equivalence_bonus
        self.feature_dim = feature_dim
        self.render_mode = render_mode
        
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
        
        # Define action space
        num_qubits = self.initial_state.qiskit_circuit.num_qubits
        num_actions = CircuitAction.get_num_actions(num_qubits, self.available_gates)
        self.action_space = spaces.Discrete(num_actions)
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(feature_dim,), dtype=np.float32
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
    
    def step(self, action_index):
        """
        Take a step in the environment by applying an action.
        
        Args:
            action_index (int): Index of the action to take.
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Convert action index to CircuitAction
        num_qubits = self.state.qiskit_circuit.num_qubits
        action = CircuitAction.from_index(action_index, self.action_space.n, num_qubits, self.available_gates)
        
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
        depth_ratio = original_state.depth / max(1, state.depth)
        depth_reward = self.depth_weight * (depth_ratio - 1.0)
        
        # Calculate cost component
        if state.cost is not None and original_state.cost is not None:
            cost_ratio = original_state.cost / max(1, state.cost)
            cost_reward = self.cost_weight * (cost_ratio - 1.0)
        else:
            cost_reward = 0.0
        
        # Calculate error component
        if state.error is not None and original_state.error is not None:
            error_ratio = (1.0 - state.error) / max(0.01, 1.0 - original_state.error)
            error_reward = self.error_weight * (error_ratio - 1.0)
        else:
            error_reward = 0.0
        
        # Calculate equivalence bonus
        try:
            if state.is_equivalent_to(original_state):
                equivalence_bonus = self.equivalence_bonus
            else:
                equivalence_bonus = 0.0
        except Exception:
            # If equivalence check fails, assume not equivalent
            equivalence_bonus = 0.0
        
        # Combine rewards
        total_reward = depth_reward + cost_reward + error_reward + equivalence_bonus
        
        return total_reward
    
    def _get_observation(self):
        """
        Get the current observation.
        
        Returns:
            np.ndarray: The observation.
        """
        return self.state.get_feature_vector(self.feature_dim)
    
    def _get_info(self):
        """
        Get information about the current state.
        
        Returns:
            dict: Information about the current state.
        """
        return {
            'depth': self.state.depth,
            'gate_count': self.state.gate_count,
            'two_qubit_gate_count': self.state.two_qubit_gate_count,
            'cost': self.state.cost,
            'error': self.state.error,
            'steps': self.steps,
            'best_depth': self.best_state.depth,
            'best_gate_count': self.best_state.gate_count,
            'best_cost': self.best_state.cost,
            'best_error': self.best_state.error,
            'best_reward': self.best_reward
        }
    
    def render(self):
        """
        Render the environment.
        
        Returns:
            None
        """
        if self.render_mode == 'human':
            print(f"Step: {self.steps}")
            print(f"Circuit depth: {self.state.depth}")
            print(f"Gate count: {self.state.gate_count}")
            print(f"Two-qubit gate count: {self.state.two_qubit_gate_count}")
            print(f"Circuit cost: {self.state.cost}")
            print(f"Circuit error: {self.state.error}")
            print(f"Best reward: {self.best_reward}")
            print(f"Circuit:\n{self.state.qiskit_circuit}")
            print("-" * 50)
    
    def close(self):
        """Close the environment."""
        pass
    
    def get_best_circuit(self):
        """
        Get the best circuit found so far.
        
        Returns:
            The best circuit.
        """
        if self.best_state is not None:
            return self.best_state.circuit
        return self.state.circuit
    
    def _generate_random_circuit(self, num_qubits, num_gates):
        """
        Generate a random quantum circuit.
        
        Args:
            num_qubits (int): Number of qubits.
            num_gates (int): Number of gates.
            
        Returns:
            A Qiskit QuantumCircuit.
        """
        circuit = QuantumCircuit(num_qubits)
        
        # Available gates
        single_qubit_gates = ['x', 'y', 'z', 'h', 'rx', 'ry', 'rz']
        two_qubit_gates = ['cx', 'cz', 'swap']
        
        # Add random gates
        for _ in range(num_gates):
            # Randomly choose between single-qubit and two-qubit gates
            if np.random.random() < 0.7 or num_qubits == 1:
                # Add a single-qubit gate
                gate = np.random.choice(single_qubit_gates)
                qubit = np.random.randint(0, num_qubits)
                
                if gate in ['rx', 'ry', 'rz']:
                    # Rotation gate with a parameter
                    angle = np.random.random() * 2 * np.pi
                    getattr(circuit, gate)(angle, qubit)
                else:
                    # Gate without parameters
                    getattr(circuit, gate)(qubit)
            else:
                # Add a two-qubit gate
                gate = np.random.choice(two_qubit_gates)
                control = np.random.randint(0, num_qubits)
                target = np.random.randint(0, num_qubits)
                
                # Make sure control and target are different
                while target == control:
                    target = np.random.randint(0, num_qubits)
                
                if gate == 'cx':
                    circuit.cx(control, target)
                elif gate == 'cz':
                    circuit.cz(control, target)
                elif gate == 'swap':
                    circuit.swap(control, target)
        
        return circuit
