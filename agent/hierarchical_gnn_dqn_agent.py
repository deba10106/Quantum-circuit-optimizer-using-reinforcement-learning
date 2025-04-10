"""
Hierarchical GNN-based DQN Agent
============================

This module implements a hierarchical GNN-based DQN agent for quantum circuit optimization,
combining the hierarchical action space with GNN-based state representation.
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
from agent.gnn_model import GNNEncoder, GNNQNetwork
from environment.hierarchical_action import HierarchicalCircuitAction, ActionCategory

# Define a transition tuple for the replay buffer
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward', 'done'))

class HierarchicalGNNDQNAgent(RLAgent):
    """
    A hierarchical GNN-based DQN agent for quantum circuit optimization.
    
    This agent uses a hierarchical action space decomposition and GNN-based
    state representation to optimize quantum circuits.
    """
    
    def __init__(self,
                 node_feature_dim,
                 num_categories,
                 num_actions_per_category,
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
            num_actions_per_category (list): List of number of actions per category.
            hidden_dim (int): Hidden dimension for the networks.
            learning_rate (float): Learning rate for the optimizer.
            gamma (float): Discount factor.
            epsilon_start (float): Starting value of epsilon for epsilon-greedy exploration.
            epsilon_end (float): Minimum value of epsilon.
            epsilon_decay (float): Decay rate of epsilon.
            buffer_size (int): Size of the replay buffer.
            batch_size (int): Batch size for training.
            target_update (int): Frequency of target network updates.
            gnn_type (str): Type of GNN to use ('gcn' or 'gat').
            num_gnn_layers (int): Number of GNN layers.
            device (str): Device to use ('cpu', 'cuda', or 'auto').
        """
        super().__init__(
            state_dim=node_feature_dim,  # Use node_feature_dim as state_dim
            action_dim=sum(num_actions_per_category),  # Total number of actions
            device=device
        )
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Store parameters
        self.node_feature_dim = node_feature_dim
        self.num_categories = num_categories
        self.num_actions_per_category = num_actions_per_category
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update = target_update
        self.gnn_type = gnn_type
        self.num_gnn_layers = num_gnn_layers
        
        # Create the GNN encoder
        self.encoder = GNNEncoder(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            gnn_type=gnn_type,
            num_layers=num_gnn_layers
        ).to(self.device)
        
        # Create the category Q-network
        self.category_q_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_categories)
        ).to(self.device)
        
        # Create the target category Q-network
        self.target_category_q_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_categories)
        ).to(self.device)
        
        # Create the action Q-networks (one for each category)
        self.action_q_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_actions)
            ).to(self.device)
            for num_actions in num_actions_per_category
        ])
        
        # Create the target action Q-networks
        self.target_action_q_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_actions)
            ).to(self.device)
            for num_actions in num_actions_per_category
        ])
        
        # Initialize target networks with the same weights
        self.target_category_q_network.load_state_dict(self.category_q_network.state_dict())
        for i in range(len(self.action_q_networks)):
            self.target_action_q_networks[i].load_state_dict(self.action_q_networks[i].state_dict())
        
        # Create optimizers
        self.category_optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.category_q_network.parameters()),
            lr=learning_rate
        )
        
        self.action_optimizers = [
            optim.Adam(
                list(self.encoder.parameters()) + list(self.action_q_networks[i].parameters()),
                lr=learning_rate
            )
            for i in range(len(self.action_q_networks))
        ]
        
        # Create replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Initialize step counter
        self.steps = 0
        
    def select_action(self, state, deterministic=False):
        """
        Select an action using the hierarchical action space.
        
        Args:
            state: The current state (node_features, edge_indices).
            deterministic (bool): Whether to select actions deterministically.
            
        Returns:
            tuple: (category_index, action_index) or HierarchicalCircuitAction
        """
        node_features, edge_indices = state
        
        # Convert to tensors
        node_features = torch.FloatTensor(node_features).to(self.device)
        edge_indices = torch.LongTensor(edge_indices).to(self.device)
        
        # Get the state embedding
        with torch.no_grad():
            state_embedding = self.encoder(node_features, edge_indices)
        
        # Epsilon-greedy exploration for category selection
        if not deterministic and random.random() < self.epsilon:
            # Random category
            category_index = random.randint(0, self.num_categories - 1)
        else:
            # Select category with highest Q-value
            with torch.no_grad():
                category_q_values = self.category_q_network(state_embedding)
                category_index = category_q_values.argmax().item()
        
        # Epsilon-greedy exploration for action selection
        if not deterministic and random.random() < self.epsilon:
            # Random action within the selected category
            action_index = random.randint(0, self.num_actions_per_category[category_index] - 1)
        else:
            # Select action with highest Q-value within the selected category
            with torch.no_grad():
                action_q_values = self.action_q_networks[category_index](state_embedding)
                action_index = action_q_values.argmax().item()
        
        # Return the hierarchical action
        return (category_index, action_index)
    
    def update(self, state, action, next_state, reward, done):
        """
        Update the agent with a new experience.
        
        Args:
            state: The current state (node_features, edge_indices).
            action: The action taken (category_index, action_index).
            next_state: The next state (node_features, edge_indices).
            reward (float): The reward received.
            done (bool): Whether the episode is done.
        """
        # Store the transition in the replay buffer
        self.replay_buffer.append(Transition(state, action, next_state, reward, done))
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Increment step counter
        self.steps += 1
        
        # Only update if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample a batch from the replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        # Unpack the batch
        states, actions, next_states, rewards, dones = zip(*batch)
        
        # Train the networks
        self._train_category_network(states, actions, next_states, rewards, dones)
        self._train_action_networks(states, actions, next_states, rewards, dones)
        
        # Update target networks
        if self.steps % self.target_update == 0:
            self.target_category_q_network.load_state_dict(self.category_q_network.state_dict())
            for i in range(len(self.action_q_networks)):
                self.target_action_q_networks[i].load_state_dict(self.action_q_networks[i].state_dict())
    
    def _train_category_network(self, states, actions, next_states, rewards, dones):
        """
        Train the category Q-network.
        
        Args:
            states: Batch of states.
            actions: Batch of actions.
            next_states: Batch of next states.
            rewards: Batch of rewards.
            dones: Batch of done flags.
        """
        # Extract category indices from actions
        category_indices = []
        for action in actions:
            # Check if action is a HierarchicalCircuitAction object
            if hasattr(action, 'category'):
                # Adjust for 1-indexed enum
                category_indices.append(action.category.value - 1)
            else:
                # If it's not a proper action object, use a default value
                category_indices.append(0)
        
        # Process each state and get embeddings
        state_embeddings = []
        for state in states:
            node_features, edge_indices = state
            node_features = torch.FloatTensor(node_features).to(self.device)
            edge_indices = torch.LongTensor(edge_indices).to(self.device)
            state_embedding = self.encoder(node_features, edge_indices)
            state_embeddings.append(state_embedding)
        
        state_embeddings = torch.stack(state_embeddings)
        
        # Get Q-values for the selected categories
        category_q_values = self.category_q_network(state_embeddings)
        
        # Check if we have any actions before proceeding
        if len(category_indices) == 0:
            return 0.0
            
        # Convert category_indices to tensor
        category_indices_tensor = torch.tensor(category_indices, dtype=torch.long, device=self.device)
        
        # Handle the case where category_q_values is 3D [batch_size, 1, num_categories]
        if category_q_values.dim() == 3:
            # Reshape to [batch_size, num_categories]
            batch_size, _, num_categories = category_q_values.shape
            category_q_values = category_q_values.reshape(batch_size, num_categories)
        
        # Ensure indices are in the correct shape for gather
        category_indices_tensor = category_indices_tensor.unsqueeze(-1)
        
        try:
            # Gather Q-values for the selected categories
            category_q_values_selected = category_q_values.gather(1, category_indices_tensor)
            category_q_values_selected = category_q_values_selected.squeeze(-1)
        except RuntimeError:
            # If gather fails, use a different approach
            # Get the Q-values for each batch item at its corresponding category index
            batch_indices = torch.arange(len(category_indices), device=self.device)
            category_q_values_selected = category_q_values[batch_indices, category_indices_tensor.squeeze(-1)]
            
        # Process each next state and get embeddings
        next_state_embeddings = []
        for next_state in next_states:
            if next_state is None:  # Terminal state
                next_state_embeddings.append(torch.zeros(self.hidden_dim).to(self.device))
                continue
                
            node_features, edge_indices = next_state
            node_features = torch.FloatTensor(node_features).to(self.device)
            edge_indices = torch.LongTensor(edge_indices).to(self.device)
            next_state_embedding = self.encoder(node_features, edge_indices)
            next_state_embeddings.append(next_state_embedding)
            
        next_state_embeddings = torch.stack(next_state_embeddings)
        
        # Get max Q-values for next states
        next_category_q_values = self.target_category_q_network(next_state_embeddings)
        
        # Handle the case where next_category_q_values is 3D
        if next_category_q_values.dim() == 3:
            # Reshape to [batch_size, num_categories]
            batch_size, _, num_categories = next_category_q_values.shape
            next_category_q_values = next_category_q_values.reshape(batch_size, num_categories)
            
        next_category_q_values, _ = next_category_q_values.max(dim=1)
        
        # Compute target Q-values
        target_q_values = torch.FloatTensor(rewards).to(self.device) + \
                         self.gamma * next_category_q_values * (1 - torch.FloatTensor(dones).to(self.device))
        
        # Compute loss
        try:
            loss = F.mse_loss(category_q_values_selected, target_q_values.detach())
            
            # Optimize
            self.category_optimizer.zero_grad()
            loss.backward()
            self.category_optimizer.step()
            
            return loss.item()
        except RuntimeError as e:
            # If loss computation fails, return a default value
            return 0.0
    
    def _train_action_networks(self, states, actions, next_states, rewards, dones):
        """
        Train the action Q-networks.
        
        Args:
            states: Batch of states.
            actions: Batch of actions.
            next_states: Batch of next states.
            rewards: Batch of rewards.
            dones: Batch of done flags.
        """
        # Convert to tensors
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Group actions by category
        category_actions = {}
        for i, action in enumerate(actions):
            # Check if action is a HierarchicalCircuitAction object
            if hasattr(action, 'category'):
                category_index = action.category.value - 1  # Adjust for 1-indexed enum
                action_type_index = action.action_type.value - 1  # Assuming action_type is also an enum
                
                if category_index not in category_actions:
                    category_actions[category_index] = []
                category_actions[category_index].append((i, action_type_index))
            else:
                # Skip invalid actions
                continue
        
        # Process each state and get embeddings
        state_embeddings = []
        for state in states:
            node_features, edge_indices = state
            node_features = torch.FloatTensor(node_features).to(self.device)
            edge_indices = torch.LongTensor(edge_indices).to(self.device)
            state_embedding = self.encoder(node_features, edge_indices)
            state_embeddings.append(state_embedding)
        
        state_embeddings = torch.stack(state_embeddings)
        
        # Process each next state and get embeddings
        next_state_embeddings = []
        for next_state in next_states:
            if next_state is None:  # Terminal state
                next_state_embeddings.append(torch.zeros(self.hidden_dim).to(self.device))
                continue
                
            node_features, edge_indices = next_state
            node_features = torch.FloatTensor(node_features).to(self.device)
            edge_indices = torch.LongTensor(edge_indices).to(self.device)
            next_state_embedding = self.encoder(node_features, edge_indices)
            next_state_embeddings.append(next_state_embedding)
            
        next_state_embeddings = torch.stack(next_state_embeddings)
        
        # Train each action network
        for category_index, action_data in category_actions.items():
            if category_index >= len(self.action_q_networks):
                continue  # Skip if category index is out of range
                
            # Get indices and action indices for this category
            indices, action_indices = zip(*action_data)
            indices = torch.LongTensor(indices).to(self.device)
            action_indices = torch.LongTensor(action_indices).to(self.device)
            
            # Get state embeddings for this category
            category_state_embeddings = state_embeddings[indices]
            
            # Get Q-values for this category
            action_q_values = self.action_q_networks[category_index](category_state_embeddings)
            
            # Handle the case where action_q_values is 3D
            if action_q_values.dim() == 3:
                # Reshape to [batch_size, num_actions]
                batch_size, _, num_actions = action_q_values.shape
                action_q_values = action_q_values.reshape(batch_size, num_actions)
            
            # Ensure action_indices have the right shape for gather
            action_indices = action_indices.unsqueeze(-1)
            
            try:
                # Gather Q-values for the selected actions
                action_q_values_selected = action_q_values.gather(1, action_indices).squeeze(-1)
                
                # Get next state embeddings for this category
                category_next_state_embeddings = next_state_embeddings[indices]
                
                # Get Q-values for next states
                next_action_q_values = self.target_action_q_networks[category_index](category_next_state_embeddings)
                
                # Handle the case where next_action_q_values is 3D
                if next_action_q_values.dim() == 3:
                    # Reshape to [batch_size, num_actions]
                    batch_size, _, num_actions = next_action_q_values.shape
                    next_action_q_values = next_action_q_values.reshape(batch_size, num_actions)
                
                # Get max Q-values for next states
                next_action_q_values, _ = next_action_q_values.max(dim=1)
                
                # Compute target Q-values
                category_rewards = rewards[indices]
                category_dones = dones[indices]
                target_q_values = category_rewards + self.gamma * next_action_q_values * (1 - category_dones)
                
                # Compute loss
                loss = F.mse_loss(action_q_values_selected, target_q_values.detach())
                
                # Optimize
                self.action_optimizers[category_index].zero_grad()
                loss.backward()
                self.action_optimizers[category_index].step()
            except (RuntimeError, IndexError) as e:
                # Skip this category if there's an error
                continue
    
    def train(self, env, num_steps):
        """
        Train the agent on an environment.
        
        Args:
            env: The environment to train on.
            num_steps (int): Number of steps to train for.
            
        Returns:
            dict: Training results.
        """
        # Initialize results
        results = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_losses': []
        }
        
        # Initialize episode variables
        episode_reward = 0
        episode_length = 0
        episode_losses = []
        
        # Reset the environment
        state, _ = env.reset()
        
        # Training loop
        for step in range(num_steps):
            # Select an action
            action = self.select_action(state)
            
            # Take a step in the environment
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Update the agent
            self.update(state, action, next_state, reward, done or truncated)
            
            # Update episode variables
            episode_reward += reward
            episode_length += 1
            
            # Move to the next state
            state = next_state
            
            # Check if episode is done
            if done or truncated:
                # Record episode results
                results['episode_rewards'].append(episode_reward)
                results['episode_lengths'].append(episode_length)
                if episode_losses:
                    results['episode_losses'].append(np.mean(episode_losses))
                
                # Reset episode variables
                episode_reward = 0
                episode_length = 0
                episode_losses = []
                
                # Reset the environment
                state, _ = env.reset()
                
                # Print progress
                if (step + 1) % 100 == 0:
                    print(f"Step {step + 1}/{num_steps}, "
                          f"Episode {len(results['episode_rewards'])}, "
                          f"Avg Reward: {np.mean(results['episode_rewards'][-10:]):.2f}, "
                          f"Epsilon: {self.epsilon:.2f}")
        
        return results
    
    def save(self, path):
        """
        Save the agent to a file.
        
        Args:
            path (str): Path to save the agent to.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the agent
        torch.save({
            'encoder': self.encoder.state_dict(),
            'category_q_network': self.category_q_network.state_dict(),
            'action_q_networks': [net.state_dict() for net in self.action_q_networks],
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
    
    def load(self, path):
        """
        Load the agent from a file.
        
        Args:
            path (str): Path to load the agent from.
        """
        # Load the agent
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load the networks
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.category_q_network.load_state_dict(checkpoint['category_q_network'])
        for i, state_dict in enumerate(checkpoint['action_q_networks']):
            self.action_q_networks[i].load_state_dict(state_dict)
        
        # Update target networks
        self.target_category_q_network.load_state_dict(self.category_q_network.state_dict())
        for i in range(len(self.action_q_networks)):
            self.target_action_q_networks[i].load_state_dict(self.action_q_networks[i].state_dict())
        
        # Load other variables
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
