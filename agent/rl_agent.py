"""
Reinforcement Learning Agent
==========================

This module provides the base class for reinforcement learning agents
used in quantum circuit optimization.
"""

import os
import numpy as np
import torch
from abc import ABC, abstractmethod

class RLAgent(ABC):
    """
    Abstract base class for reinforcement learning agents.
    
    This class defines the interface that all RL agents must implement.
    """
    
    def __init__(self, state_dim, action_dim, device='auto'):
        """
        Initialize the RL agent.
        
        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            device (str): Device to run the model on ('cpu', 'cuda', or 'auto').
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
    
    @abstractmethod
    def select_action(self, state, deterministic=False):
        """
        Select an action based on the current state.
        
        Args:
            state: The current state.
            deterministic (bool): Whether to select the action deterministically.
            
        Returns:
            The selected action.
        """
        pass
    
    @abstractmethod
    def train(self, env, num_steps, log_interval=1000):
        """
        Train the agent on an environment.
        
        Args:
            env: The environment to train on.
            num_steps (int): Number of steps to train for.
            log_interval (int): Interval for logging.
            
        Returns:
            dict: Training statistics.
        """
        pass
    
    @abstractmethod
    def save(self, path):
        """
        Save the agent to a file.
        
        Args:
            path (str): Path to save the agent to.
        """
        pass
    
    @abstractmethod
    def load(self, path):
        """
        Load the agent from a file.
        
        Args:
            path (str): Path to load the agent from.
        """
        pass
    
    def preprocess_state(self, state):
        """
        Preprocess a state for input to the model.
        
        Args:
            state: The state to preprocess.
            
        Returns:
            torch.Tensor: The preprocessed state.
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        elif isinstance(state, torch.Tensor):
            state = state.to(self.device)
        else:
            raise ValueError(f"Unsupported state type: {type(state)}")
            
        # Add batch dimension if needed
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        return state
    
    def evaluate(self, env, num_episodes=10, render=False):
        """
        Evaluate the agent on an environment.
        
        Args:
            env: The environment to evaluate on.
            num_episodes (int): Number of episodes to evaluate for.
            render (bool): Whether to render the environment.
            
        Returns:
            dict: Evaluation statistics.
        """
        total_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            truncated = False
            total_reward = 0
            steps = 0
            
            while not (done or truncated):
                action = self.select_action(state, deterministic=True)
                state, reward, done, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                
                if render:
                    env.render()
            
            total_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Reward = {total_reward:.4f}, Length = {steps}")
        
        # Calculate statistics
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        mean_length = np.mean(episode_lengths)
        std_length = np.std(episode_lengths)
        
        print(f"Evaluation results:")
        print(f"  Mean reward: {mean_reward:.4f} ± {std_reward:.4f}")
        print(f"  Mean length: {mean_length:.2f} ± {std_length:.2f}")
        
        return {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_length': mean_length,
            'std_length': std_length,
            'rewards': total_rewards,
            'lengths': episode_lengths
        }
    
    def optimize_circuit(self, env, circuit, max_steps=100, render=False):
        """
        Optimize a quantum circuit using the trained agent.
        
        Args:
            env: The environment to use for optimization.
            circuit: The circuit to optimize.
            max_steps (int): Maximum number of optimization steps.
            render (bool): Whether to render the environment.
            
        Returns:
            tuple: (optimized_circuit, optimization_info)
        """
        # Set the initial circuit
        env.initial_circuit = circuit
        env.max_steps = max_steps
        
        # Reset the environment
        state, _ = env.reset()
        done = False
        truncated = False
        steps = 0
        
        # Optimization loop
        while not (done or truncated):
            # Select action
            action = self.select_action(state, deterministic=True)
            
            # Take step
            state, reward, done, truncated, info = env.step(action)
            steps += 1
            
            if render:
                env.render()
        
        # Get the best circuit
        optimized_circuit = env.get_best_circuit()
        
        # Gather optimization info
        optimization_info = {
            'initial_depth': env.original_state.depth,
            'initial_gate_count': env.original_state.gate_count,
            'initial_cost': env.original_state.cost,
            'initial_error': env.original_state.error,
            'optimized_depth': env.best_state.depth,
            'optimized_gate_count': env.best_state.gate_count,
            'optimized_cost': env.best_state.cost,
            'optimized_error': env.best_state.error,
            'depth_reduction': env.original_state.depth - env.best_state.depth,
            'gate_count_reduction': env.original_state.gate_count - env.best_state.gate_count,
            'cost_reduction': env.original_state.cost - env.best_state.cost if env.original_state.cost and env.best_state.cost else None,
            'error_reduction': env.original_state.error - env.best_state.error if env.original_state.error and env.best_state.error else None,
            'steps': steps,
            'best_reward': env.best_reward
        }
        
        return optimized_circuit, optimization_info
