"""
Proximal Policy Optimization Agent
================================

This module provides a PPO agent for quantum circuit optimization.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from agent.rl_agent import RLAgent

class ActorCritic(nn.Module):
    """Neural network for the actor-critic architecture."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Initialize the actor-critic network.
        
        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Dimension of the hidden layers.
        """
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input state.
            
        Returns:
            tuple: (action_probs, state_value)
        """
        features = self.feature_extractor(x)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        
        return action_probs, state_value
    
    def act(self, state, deterministic=False):
        """
        Select an action based on the current state.
        
        Args:
            state (torch.Tensor): The current state.
            deterministic (bool): Whether to select the action deterministically.
            
        Returns:
            tuple: (action, log_prob, entropy)
        """
        action_probs, _ = self(state)
        
        # Create a distribution
        dist = Categorical(action_probs)
        
        if deterministic:
            # Select the most probable action
            action = torch.argmax(action_probs, dim=-1)
        else:
            # Sample an action
            action = dist.sample()
        
        # Get log probability and entropy
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy
    
    def evaluate(self, state, action):
        """
        Evaluate an action in the given state.
        
        Args:
            state (torch.Tensor): The state.
            action (torch.Tensor): The action.
            
        Returns:
            tuple: (log_prob, state_value, entropy)
        """
        action_probs, state_value = self(state)
        
        # Create a distribution
        dist = Categorical(action_probs)
        
        # Get log probability and entropy
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_prob, state_value, entropy

class PPOAgent(RLAgent):
    """
    Proximal Policy Optimization agent for quantum circuit optimization.
    
    This agent uses the PPO algorithm to learn a policy for optimizing quantum circuits.
    """
    
    def __init__(self, state_dim, action_dim, 
                 hidden_dim=128, 
                 learning_rate=3e-4, 
                 gamma=0.99, 
                 gae_lambda=0.95, 
                 clip_param=0.2, 
                 value_coef=0.5, 
                 entropy_coef=0.01, 
                 max_grad_norm=0.5, 
                 ppo_epochs=10, 
                 batch_size=64, 
                 device='auto'):
        """
        Initialize the PPO agent.
        
        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Dimension of the hidden layers.
            learning_rate (float): Learning rate for the optimizer.
            gamma (float): Discount factor.
            gae_lambda (float): GAE lambda parameter.
            clip_param (float): PPO clipping parameter.
            value_coef (float): Value function coefficient.
            entropy_coef (float): Entropy coefficient.
            max_grad_norm (float): Maximum gradient norm.
            ppo_epochs (int): Number of PPO epochs.
            batch_size (int): Batch size for training.
            device (str): Device to run the model on ('cpu', 'cuda', or 'auto').
        """
        super(PPOAgent, self).__init__(state_dim, action_dim, device)
        
        # Store parameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # Create actor-critic network
        self.network = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Create optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    
    def select_action(self, state, deterministic=False):
        """
        Select an action based on the current state.
        
        Args:
            state: The current state.
            deterministic (bool): Whether to select the action deterministically.
            
        Returns:
            The selected action.
        """
        # Preprocess state
        state = self.preprocess_state(state)
        
        # Select action
        with torch.no_grad():
            action, _, _ = self.network.act(state, deterministic)
        
        return action.item()
    
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
        # Initialize statistics
        episode_rewards = []
        episode_lengths = []
        episode_reward = 0
        episode_length = 0
        
        # Reset environment
        state, _ = env.reset()
        
        # Training loop
        steps_done = 0
        while steps_done < num_steps:
            # Collect rollout
            rollout = self._collect_rollout(env, 2048)
            steps_done += len(rollout['states'])
            
            # Update statistics
            episode_rewards.extend(rollout['episode_rewards'])
            episode_lengths.extend(rollout['episode_lengths'])
            
            # Update policy
            self._update_policy(rollout)
            
            # Log progress
            if len(episode_rewards) % log_interval == 0:
                avg_reward = np.mean(episode_rewards[-log_interval:])
                avg_length = np.mean(episode_lengths[-log_interval:])
                print(f"Step {steps_done}/{num_steps}: "
                      f"Avg. Reward = {avg_reward:.4f}, "
                      f"Avg. Length = {avg_length:.2f}")
        
        # Calculate final statistics
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        avg_length = np.mean(episode_lengths)
        std_length = np.std(episode_lengths)
        
        print(f"Training completed:")
        print(f"  Mean reward: {avg_reward:.4f} ± {std_reward:.4f}")
        print(f"  Mean length: {avg_length:.2f} ± {std_length:.2f}")
        
        return {
            'mean_reward': avg_reward,
            'std_reward': std_reward,
            'mean_length': avg_length,
            'std_length': std_length,
            'rewards': episode_rewards,
            'lengths': episode_lengths
        }
    
    def _collect_rollout(self, env, rollout_length):
        """
        Collect a rollout from the environment.
        
        Args:
            env: The environment to collect from.
            rollout_length (int): Length of the rollout.
            
        Returns:
            dict: Rollout data.
        """
        # Initialize rollout data
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []
        
        # Initialize episode statistics
        episode_rewards = []
        episode_lengths = []
        episode_reward = 0
        episode_length = 0
        
        # Reset environment
        state, _ = env.reset()
        
        # Collect rollout
        for _ in range(rollout_length):
            # Preprocess state
            state_tensor = self.preprocess_state(state)
            
            # Get action and value
            with torch.no_grad():
                action, log_prob, _ = self.network.act(state_tensor)
                _, value, _ = self.network.evaluate(state_tensor, action)
            
            # Take step in environment
            next_state, reward, done, truncated, _ = env.step(action.item())
            
            # Store data
            states.append(state)
            actions.append(action.item())
            log_probs.append(log_prob.item())
            rewards.append(reward)
            dones.append(done or truncated)
            values.append(value.item())
            
            # Update episode statistics
            episode_reward += reward
            episode_length += 1
            
            # Check if episode is done
            if done or truncated:
                # Store episode statistics
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # Reset episode statistics
                episode_reward = 0
                episode_length = 0
                
                # Reset environment
                state, _ = env.reset()
            else:
                # Move to next state
                state = next_state
        
        # Calculate advantages and returns
        returns, advantages = self._compute_advantages_and_returns(
            rewards, values, dones, self.gamma, self.gae_lambda
        )
        
        # Return rollout data
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'log_probs': np.array(log_probs),
            'rewards': np.array(rewards),
            'dones': np.array(dones),
            'values': np.array(values),
            'returns': returns,
            'advantages': advantages,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }
    
    def _compute_advantages_and_returns(self, rewards, values, dones, gamma, gae_lambda):
        """
        Compute advantages and returns using Generalized Advantage Estimation (GAE).
        
        Args:
            rewards (list): List of rewards.
            values (list): List of state values.
            dones (list): List of done flags.
            gamma (float): Discount factor.
            gae_lambda (float): GAE lambda parameter.
            
        Returns:
            tuple: (returns, advantages)
        """
        # Initialize arrays
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        
        # Initialize variables
        gae = 0
        next_value = 0  # Assume zero value for terminal state
        
        # Compute advantages and returns in reverse order
        for t in reversed(range(len(rewards))):
            # Compute delta
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0.0  # Assume zero value for terminal state
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            
            # Compute GAE
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            
            # Compute returns
            returns[t] = advantages[t] + values[t]
        
        return returns, advantages
    
    def _update_policy(self, rollout):
        """
        Update the policy using PPO.
        
        Args:
            rollout (dict): Rollout data.
        """
        # Extract data
        states = torch.FloatTensor(rollout['states']).to(self.device)
        actions = torch.LongTensor(rollout['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(rollout['log_probs']).to(self.device)
        returns = torch.FloatTensor(rollout['returns']).to(self.device)
        advantages = torch.FloatTensor(rollout['advantages']).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.ppo_epochs):
            # Generate random indices
            indices = torch.randperm(states.size(0))
            
            # Update in batches
            for start_idx in range(0, states.size(0), self.batch_size):
                # Get batch indices
                idx = indices[start_idx:start_idx + self.batch_size]
                
                # Get batch data
                state_batch = states[idx]
                action_batch = actions[idx]
                old_log_prob_batch = old_log_probs[idx]
                return_batch = returns[idx]
                advantage_batch = advantages[idx]
                
                # Evaluate actions
                log_probs, state_values, entropy = self.network.evaluate(state_batch, action_batch)
                
                # Calculate ratios
                ratios = torch.exp(log_probs - old_log_prob_batch)
                
                # Calculate surrogate losses
                surr1 = ratios * advantage_batch
                surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage_batch
                
                # Calculate losses
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(state_values.squeeze(-1), return_batch)
                entropy_loss = -entropy.mean()
                
                # Calculate total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Perform update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
    
    def save(self, path):
        """
        Save the agent to a file.
        
        Args:
            path (str): Path to save the agent to.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model and parameters
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_param': self.clip_param,
            'value_coef': self.value_coef,
            'entropy_coef': self.entropy_coef,
            'max_grad_norm': self.max_grad_norm,
            'ppo_epochs': self.ppo_epochs,
            'batch_size': self.batch_size
        }, path)
        
        print(f"Agent saved to {path}")
    
    def load(self, path):
        """
        Load the agent from a file.
        
        Args:
            path (str): Path to load the agent from.
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load parameters
        self.state_dim = checkpoint['state_dim']
        self.action_dim = checkpoint['action_dim']
        self.gamma = checkpoint['gamma']
        self.gae_lambda = checkpoint['gae_lambda']
        self.clip_param = checkpoint['clip_param']
        self.value_coef = checkpoint['value_coef']
        self.entropy_coef = checkpoint['entropy_coef']
        self.max_grad_norm = checkpoint['max_grad_norm']
        self.ppo_epochs = checkpoint['ppo_epochs']
        self.batch_size = checkpoint['batch_size']
        
        # Create network
        hidden_dim = 128  # Assuming default hidden dim
        self.network = ActorCritic(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        
        # Load state dicts
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Agent loaded from {path}")
