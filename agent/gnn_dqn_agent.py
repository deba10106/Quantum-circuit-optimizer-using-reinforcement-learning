"""
GNN-based DQN Agent
=================

This module provides a DQN agent with GNN-based state representation for quantum circuit optimization,
implementing the neural architecture from the Quarl paper.
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
from agent.gnn_model import GNNQNetwork, HierarchicalGNNQNetwork
from environment.hierarchical_action import ActionCategory, HierarchicalCircuitAction

# Define a transition tuple for the replay buffer
Transition = namedtuple('Transition', 
                        ('node_features', 'edge_index', 'action', 'next_node_features', 
                         'next_edge_index', 'reward', 'done'))

class GNNReplayBuffer:
    """A replay buffer for storing and sampling transitions with graph data."""
    
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

class GNNDQNAgent(RLAgent):
    """
    Deep Q-Network agent with GNN-based state representation for quantum circuit optimization.
    This implements the neural architecture from the Quarl paper.
    """
    
    def __init__(self, node_feature_dim, action_dim, 
                 hidden_dim=64, 
                 learning_rate=1e-3, 
                 gamma=0.99, 
                 epsilon_start=1.0, 
                 epsilon_end=0.1, 
                 epsilon_decay=0.995, 
                 buffer_size=10000, 
                 batch_size=32, 
                 target_update=10, 
                 gnn_type='gcn',
                 num_gnn_layers=3,
                 device='auto'):
        """
        Initialize the GNN-based DQN agent.
        
        Args:
            node_feature_dim (int): Dimension of node features.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Dimension of hidden layers.
            learning_rate (float): Learning rate for the optimizer.
            gamma (float): Discount factor.
            epsilon_start (float): Initial exploration rate.
            epsilon_end (float): Final exploration rate.
            epsilon_decay (float): Decay rate for exploration.
            buffer_size (int): Size of the replay buffer.
            batch_size (int): Batch size for training.
            target_update (int): Frequency of target network updates.
            gnn_type (str): Type of GNN ('gcn' or 'gat').
            num_gnn_layers (int): Number of GNN layers.
            device (str): Device to run the model on ('cpu', 'cuda', or 'auto').
        """
        super(GNNDQNAgent, self).__init__(node_feature_dim, action_dim, device)
        
        # Store parameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Create networks
        self.policy_net = GNNQNetwork(
            node_feature_dim=node_feature_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            gnn_type=gnn_type
        ).to(self.device)
        
        self.target_net = GNNQNetwork(
            node_feature_dim=node_feature_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            gnn_type=gnn_type
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Create optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Create replay buffer
        self.replay_buffer = GNNReplayBuffer(buffer_size)
        
        # Initialize step counter
        self.steps_done = 0
    
    def select_action(self, state, deterministic=False):
        """
        Select an action based on the current state.
        
        Args:
            state: The current state (graph representation).
            deterministic (bool): Whether to select the action deterministically.
            
        Returns:
            The selected action.
        """
        # Extract graph data from state
        node_features, edge_index = state
        
        # Convert to tensors
        node_features = torch.FloatTensor(node_features).to(self.device)
        edge_index = torch.LongTensor(edge_index).to(self.device)
        
        # Epsilon-greedy action selection
        if not deterministic and random.random() < self.epsilon:
            # Random action
            return random.randrange(self.action_dim)
        else:
            # Greedy action
            with torch.no_grad():
                q_values = self.policy_net(node_features, edge_index)
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
            
            # Extract graph data
            node_features, edge_index = state
            next_node_features, next_edge_index = next_state
            
            # Store transition in replay buffer
            self.replay_buffer.push(
                torch.FloatTensor(node_features),
                torch.LongTensor(edge_index),
                torch.tensor([action]),
                torch.FloatTensor(next_node_features),
                torch.LongTensor(next_edge_index),
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
                # Record episode statistics
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # Reset episode statistics
                episode_reward = 0
                episode_length = 0
                
                # Reset environment
                state, _ = env.reset()
                
            # Log progress
            if step % log_interval == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                avg_length = np.mean(episode_lengths[-100:]) if episode_lengths else 0
                print(f"Step {step}/{num_steps} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Length: {avg_length:.2f} | "
                      f"Epsilon: {self.epsilon:.2f}")
        
        # Return training statistics
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'final_epsilon': self.epsilon
        }
    
    def _optimize_model(self):
        """Perform a single optimization step."""
        if len(self.replay_buffer) < self.batch_size:
            return
            
        # Sample a batch of transitions
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Create batched graph data
        node_features_batch = torch.cat([f.unsqueeze(0) for f in batch.node_features]).to(self.device)
        edge_index_batch = torch.cat([e.unsqueeze(0) for e in batch.edge_index]).to(self.device)
        next_node_features_batch = torch.cat([f.unsqueeze(0) for f in batch.next_node_features]).to(self.device)
        next_edge_index_batch = torch.cat([e.unsqueeze(0) for e in batch.next_edge_index]).to(self.device)
        
        # Get other batch data
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        done_batch = torch.cat(batch.done).to(self.device)
        
        # Compute Q(s_t, a)
        q_values = self.policy_net(node_features_batch, edge_index_batch)
        state_action_values = q_values.gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states
        with torch.no_grad():
            next_q_values = self.target_net(next_node_features_batch, next_edge_index_batch)
            next_state_values = next_q_values.max(1)[0].unsqueeze(1)
            next_state_values[done_batch] = 0.0
            
        # Compute the expected Q values
        expected_state_action_values = reward_batch + (self.gamma * next_state_values)
        
        # Compute loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
    def save(self, path):
        """Save the agent to a file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        
    def load(self, path):
        """Load the agent from a file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']

class HierarchicalGNNDQNAgent(RLAgent):
    """
    Hierarchical DQN agent with GNN-based state representation for quantum circuit optimization.
    This implements the two-part decomposition approach from the Quarl paper.
    """
    
    def __init__(self, node_feature_dim, num_categories, num_actions_per_category,
                 hidden_dim=64, 
                 learning_rate=1e-3, 
                 gamma=0.99, 
                 epsilon_start=1.0, 
                 epsilon_end=0.1, 
                 epsilon_decay=0.995, 
                 buffer_size=10000, 
                 batch_size=32, 
                 target_update=10, 
                 gnn_type='gcn',
                 num_gnn_layers=3,
                 device='auto'):
        """
        Initialize the hierarchical GNN-based DQN agent.
        
        Args:
            node_feature_dim (int): Dimension of node features.
            num_categories (int): Number of action categories.
            num_actions_per_category (list): Number of actions for each category.
            hidden_dim (int): Dimension of hidden layers.
            learning_rate (float): Learning rate for the optimizer.
            gamma (float): Discount factor.
            epsilon_start (float): Initial exploration rate.
            epsilon_end (float): Final exploration rate.
            epsilon_decay (float): Decay rate for exploration.
            buffer_size (int): Size of the replay buffer.
            batch_size (int): Batch size for training.
            target_update (int): Frequency of target network updates.
            gnn_type (str): Type of GNN ('gcn' or 'gat').
            num_gnn_layers (int): Number of GNN layers.
            device (str): Device to run the model on ('cpu', 'cuda', or 'auto').
        """
        super(HierarchicalGNNDQNAgent, self).__init__(node_feature_dim, sum(num_actions_per_category), device)
        
        # Store parameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.num_categories = num_categories
        self.num_actions_per_category = num_actions_per_category
        
        # Create networks
        self.policy_net = HierarchicalGNNQNetwork(
            node_feature_dim=node_feature_dim,
            num_categories=num_categories,
            num_actions_per_category=num_actions_per_category,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            gnn_type=gnn_type
        ).to(self.device)
        
        self.target_net = HierarchicalGNNQNetwork(
            node_feature_dim=node_feature_dim,
            num_categories=num_categories,
            num_actions_per_category=num_actions_per_category,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            gnn_type=gnn_type
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Create optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Create replay buffer
        self.replay_buffer = GNNReplayBuffer(buffer_size)
        
        # Initialize step counter
        self.steps_done = 0
    
    def select_action(self, state, deterministic=False):
        """
        Select an action based on the current state.
        
        Args:
            state: The current state (graph representation).
            deterministic (bool): Whether to select the action deterministically.
            
        Returns:
            tuple: (category_index, action_index)
        """
        # Extract graph data from state
        node_features, edge_index = state
        
        # Convert to tensors
        node_features = torch.FloatTensor(node_features).to(self.device)
        edge_index = torch.LongTensor(edge_index).to(self.device)
        
        # Get action from the network
        return self.policy_net.get_action(
            node_features, edge_index, deterministic=deterministic
        )
    
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
            category_index, action_index, _, _ = self.select_action(state)
            
            # Convert to hierarchical action
            hierarchical_action = HierarchicalCircuitAction.from_indices(
                category_index, action_index, 
                env.action_space.n, env.num_qubits, env.available_gates
            )
            
            # Apply the action
            next_state, reward, done, truncated, _ = env.step(hierarchical_action)
            
            # Extract graph data
            node_features, edge_index = state
            next_node_features, next_edge_index = next_state
            
            # Store transition in replay buffer
            self.replay_buffer.push(
                torch.FloatTensor(node_features),
                torch.LongTensor(edge_index),
                (category_index, action_index),
                torch.FloatTensor(next_node_features),
                torch.LongTensor(next_edge_index),
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
                # Record episode statistics
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # Reset episode statistics
                episode_reward = 0
                episode_length = 0
                
                # Reset environment
                state, _ = env.reset()
                
            # Log progress
            if step % log_interval == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                avg_length = np.mean(episode_lengths[-100:]) if episode_lengths else 0
                print(f"Step {step}/{num_steps} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Length: {avg_length:.2f} | "
                      f"Epsilon: {self.epsilon:.2f}")
        
        # Return training statistics
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'final_epsilon': self.epsilon
        }
    
    def _optimize_model(self):
        """Perform a single optimization step."""
        if len(self.replay_buffer) < self.batch_size:
            return
            
        # Sample a batch of transitions
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Create batched graph data
        node_features_batch = torch.cat([f.unsqueeze(0) for f in batch.node_features]).to(self.device)
        edge_index_batch = torch.cat([e.unsqueeze(0) for e in batch.edge_index]).to(self.device)
        next_node_features_batch = torch.cat([f.unsqueeze(0) for f in batch.next_node_features]).to(self.device)
        next_edge_index_batch = torch.cat([e.unsqueeze(0) for e in batch.next_edge_index]).to(self.device)
        
        # Get other batch data
        category_batch, action_batch = zip(*batch.action)
        category_batch = torch.tensor(category_batch).to(self.device)
        action_batch = torch.tensor(action_batch).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        done_batch = torch.cat(batch.done).to(self.device)
        
        # Compute Q(s_t, a) for the category and action
        category_q_values, action_q_values = self.policy_net(node_features_batch, edge_index_batch)
        
        # Get Q-values for the selected categories and actions
        category_state_action_values = category_q_values.gather(1, category_batch.unsqueeze(1))
        
        action_state_action_values = []
        for i, (cat, act) in enumerate(zip(category_batch, action_batch)):
            action_state_action_values.append(action_q_values[cat][i, act])
        action_state_action_values = torch.stack(action_state_action_values).unsqueeze(1)
        
        # Compute combined Q-value
        state_action_values = category_state_action_values + action_state_action_values
        
        # Compute V(s_{t+1}) for all next states
        with torch.no_grad():
            next_category_q_values, next_action_q_values = self.target_net(
                next_node_features_batch, next_edge_index_batch
            )
            
            # Get best category
            next_category_values, next_category_indices = next_category_q_values.max(1)
            
            # Get best action for each category
            next_action_values = []
            for i, cat in enumerate(next_category_indices):
                next_action_values.append(next_action_q_values[cat][i].max())
            next_action_values = torch.stack(next_action_values)
            
            # Combine values
            next_state_values = (next_category_values + next_action_values).unsqueeze(1)
            next_state_values[done_batch] = 0.0
            
        # Compute the expected Q values
        expected_state_action_values = reward_batch + (self.gamma * next_state_values)
        
        # Compute loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
    def save(self, path):
        """Save the agent to a file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        
    def load(self, path):
        """Load the agent from a file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
