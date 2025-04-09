"""
Deep Q-Network Agent
==================

This module provides a DQN agent for quantum circuit optimization.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple

from agent.rl_agent import RLAgent

# Define a transition tuple for the replay buffer
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    """A simple replay buffer for storing and sampling transitions."""
    
    def __init__(self, capacity):
        """
        Initialize the replay buffer.
        
        Args:
            capacity (int): Maximum capacity of the buffer.
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        """Add a transition to the buffer."""
        self.buffer.append(Transition(*args))
    
    def sample(self, batch_size):
        """Sample a batch of transitions from the buffer."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)

class QNetwork(nn.Module):
    """Neural network for approximating the Q-function."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Initialize the Q-network.
        
        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Dimension of the hidden layers.
        """
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        """Forward pass through the network."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent(RLAgent):
    """
    Deep Q-Network agent for quantum circuit optimization.
    
    This agent uses a deep Q-network to learn a policy for optimizing quantum circuits.
    """
    
    def __init__(self, state_dim, action_dim, 
                 hidden_dim=128, 
                 learning_rate=1e-3, 
                 gamma=0.99, 
                 epsilon_start=1.0, 
                 epsilon_end=0.1, 
                 epsilon_decay=0.995, 
                 buffer_size=10000, 
                 batch_size=64, 
                 target_update=10, 
                 device='auto'):
        """
        Initialize the DQN agent.
        
        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Dimension of the hidden layers.
            learning_rate (float): Learning rate for the optimizer.
            gamma (float): Discount factor.
            epsilon_start (float): Initial exploration rate.
            epsilon_end (float): Final exploration rate.
            epsilon_decay (float): Decay rate for exploration.
            buffer_size (int): Size of the replay buffer.
            batch_size (int): Batch size for training.
            target_update (int): Frequency of target network updates.
            device (str): Device to run the model on ('cpu', 'cuda', or 'auto').
        """
        super(DQNAgent, self).__init__(state_dim, action_dim, device)
        
        # Store parameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Create networks
        self.policy_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Create optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Initialize step counter
        self.steps_done = 0
    
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
        
        # Epsilon-greedy action selection
        if not deterministic and random.random() < self.epsilon:
            # Random action
            return random.randrange(self.action_dim)
        else:
            # Greedy action
            with torch.no_grad():
                q_values = self.policy_net(state)
                return q_values.max(1)[1].item()
    
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
        for step in range(1, num_steps + 1):
            # Select and perform action
            action = self.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store transition in replay buffer
            self.replay_buffer.push(
                torch.FloatTensor(state),
                torch.tensor([action]),
                torch.FloatTensor(next_state),
                torch.tensor([reward]),
                torch.tensor([done or truncated])
            )
            
            # Move to the next state
            state = next_state
            
            # Update statistics
            episode_reward += reward
            episode_length += 1
            
            # Perform optimization step
            if len(self.replay_buffer) >= self.batch_size:
                self._optimize_model()
            
            # Update target network
            if step % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # Check if episode is done
            if done or truncated:
                # Log episode statistics
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # Reset episode statistics
                episode_reward = 0
                episode_length = 0
                
                # Reset environment
                state, _ = env.reset()
                
                # Log progress
                if len(episode_rewards) % log_interval == 0:
                    avg_reward = np.mean(episode_rewards[-log_interval:])
                    avg_length = np.mean(episode_lengths[-log_interval:])
                    print(f"Step {step}/{num_steps}: "
                          f"Avg. Reward = {avg_reward:.4f}, "
                          f"Avg. Length = {avg_length:.2f}, "
                          f"Epsilon = {self.epsilon:.4f}")
            
            # Update step counter
            self.steps_done += 1
        
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
    
    def _optimize_model(self):
        """Perform a single optimization step."""
        # Sample a batch from the replay buffer
        if len(self.replay_buffer) < self.batch_size:
            return
        
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Compute a mask of non-final states
        non_final_mask = ~torch.tensor(batch.done, dtype=torch.bool, device=self.device)
        non_final_next_states = torch.cat([s.unsqueeze(0) for s, d in zip(batch.next_state, batch.done) 
                                           if not d]).to(self.device)
        
        # Prepare batch
        state_batch = torch.cat([s.unsqueeze(0) for s in batch.state]).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device).view(-1, 1)  # Ensure action_batch is of shape [batch_size, 1]
        reward_batch = torch.cat(batch.reward).to(self.device)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if non_final_mask.any():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        
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
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'gamma': self.gamma,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'batch_size': self.batch_size,
            'target_update': self.target_update
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
        self.epsilon = checkpoint['epsilon']
        self.epsilon_end = checkpoint['epsilon_end']
        self.epsilon_decay = checkpoint['epsilon_decay']
        self.batch_size = checkpoint['batch_size']
        self.target_update = checkpoint['target_update']
        self.steps_done = checkpoint['steps_done']
        
        # Create networks
        hidden_dim = self.policy_net.fc1.out_features  # Get hidden dim from existing network
        self.policy_net = QNetwork(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        
        # Load state dicts
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Set target network to evaluation mode
        self.target_net.eval()
        
        print(f"Agent loaded from {path}")
