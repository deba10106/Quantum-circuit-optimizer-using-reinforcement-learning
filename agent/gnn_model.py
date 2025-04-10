"""
Graph Neural Network Models
=========================

This module provides GNN-based models for quantum circuit optimization,
implementing the neural architecture from the Quarl paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool

class GNNEncoder(nn.Module):
    """
    Graph Neural Network encoder for quantum circuits.
    This implements the GNN-based state representation from the Quarl paper.
    """
    
    def __init__(self, node_feature_dim, hidden_dim=64, num_layers=3, gnn_type='gcn'):
        """
        Initialize the GNN encoder.
        
        Args:
            node_feature_dim (int): Dimension of node features.
            hidden_dim (int): Dimension of hidden layers.
            num_layers (int): Number of GNN layers.
            gnn_type (str): Type of GNN ('gcn' or 'gat').
        """
        super(GNNEncoder, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        
        # Input projection
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if gnn_type == 'gcn':
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == 'gat':
                self.gnn_layers.append(GATConv(hidden_dim, hidden_dim))
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
                
        # Layer normalization
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass through the GNN encoder.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, node_feature_dim].
            edge_index (torch.Tensor): Edge indices [2, num_edges].
            batch (torch.Tensor, optional): Batch indices for multiple graphs.
            
        Returns:
            torch.Tensor: Graph embeddings.
        """
        # Input projection
        x = F.relu(self.input_proj(x))
        
        # GNN layers with residual connections and layer normalization
        for i, (gnn, norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            identity = x
            x = gnn(x, edge_index)
            x = F.relu(x)
            x = norm(x + identity)  # Residual connection
            
        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
            
        # Output projection
        x = F.relu(self.output_proj(x))
        
        return x

    def __repr__(self):
        """Return a string representation of the GNN encoder."""
        return f"GNNEncoder(node_feature_dim={self.node_feature_dim}, hidden_dim={self.hidden_dim}, output_dim={self.hidden_dim}, num_layers={self.num_layers}, gnn_type='{self.gnn_type}')"

class GNNQNetwork(nn.Module):
    """
    Graph Neural Network-based Q-Network for quantum circuit optimization.
    """
    
    def __init__(self, node_feature_dim, action_dim, hidden_dim=64, num_gnn_layers=3, gnn_type='gcn'):
        """
        Initialize the GNN Q-Network.
        
        Args:
            node_feature_dim (int): Dimension of node features.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Dimension of hidden layers.
            num_gnn_layers (int): Number of GNN layers.
            gnn_type (str): Type of GNN ('gcn' or 'gat').
        """
        super(GNNQNetwork, self).__init__()
        
        # GNN encoder
        self.encoder = GNNEncoder(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            gnn_type=gnn_type
        )
        
        # Q-value prediction layers
        self.q_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass through the GNN Q-Network.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, node_feature_dim].
            edge_index (torch.Tensor): Edge indices [2, num_edges].
            batch (torch.Tensor, optional): Batch indices for multiple graphs.
            
        Returns:
            torch.Tensor: Q-values for each action.
        """
        # Get graph embeddings
        embeddings = self.encoder(x, edge_index, batch)
        
        # Predict Q-values
        q_values = self.q_layers(embeddings)
        
        return q_values

class HierarchicalGNNQNetwork(nn.Module):
    """
    Hierarchical GNN-based Q-Network for quantum circuit optimization.
    This implements the two-part decomposition approach from the Quarl paper.
    """
    
    def __init__(self, node_feature_dim, num_categories, num_actions_per_category, 
                 hidden_dim=64, num_gnn_layers=3, gnn_type='gcn'):
        """
        Initialize the hierarchical GNN Q-Network.
        
        Args:
            node_feature_dim (int): Dimension of node features.
            num_categories (int): Number of action categories.
            num_actions_per_category (list): Number of actions for each category.
            hidden_dim (int): Dimension of hidden layers.
            num_gnn_layers (int): Number of GNN layers.
            gnn_type (str): Type of GNN ('gcn' or 'gat').
        """
        super(HierarchicalGNNQNetwork, self).__init__()
        
        self.num_categories = num_categories
        self.num_actions_per_category = num_actions_per_category
        
        # GNN encoder (shared)
        self.encoder = GNNEncoder(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            gnn_type=gnn_type
        )
        
        # Category selection network
        self.category_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_categories)
        )
        
        # Action selection networks (one per category)
        self.action_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_actions)
            )
            for num_actions in num_actions_per_category
        ])
        
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass through the hierarchical GNN Q-Network.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, node_feature_dim].
            edge_index (torch.Tensor): Edge indices [2, num_edges].
            batch (torch.Tensor, optional): Batch indices for multiple graphs.
            
        Returns:
            tuple: (category_q_values, action_q_values)
        """
        # Get graph embeddings
        embeddings = self.encoder(x, edge_index, batch)
        
        # Predict category Q-values
        category_q_values = self.category_net(embeddings)
        
        # Predict action Q-values for each category
        action_q_values = [net(embeddings) for net in self.action_nets]
        
        return category_q_values, action_q_values
    
    def get_action(self, x, edge_index, batch=None, deterministic=False):
        """
        Get the best action according to the Q-values.
        
        Args:
            x (torch.Tensor): Node features [num_nodes, node_feature_dim].
            edge_index (torch.Tensor): Edge indices [2, num_edges].
            batch (torch.Tensor, optional): Batch indices for multiple graphs.
            deterministic (bool): Whether to select actions deterministically.
            
        Returns:
            tuple: (category_index, action_index, category_q_value, action_q_value)
        """
        category_q_values, action_q_values = self.forward(x, edge_index, batch)
        
        # Select category
        if not deterministic and torch.rand(1).item() < 0.1:  # Epsilon-greedy
            category_index = torch.randint(0, self.num_categories, (1,)).item()
        else:
            category_index = category_q_values.argmax(dim=1).item()
            
        # Select action within the category
        if not deterministic and torch.rand(1).item() < 0.1:  # Epsilon-greedy
            action_index = torch.randint(0, self.num_actions_per_category[category_index], (1,)).item()
        else:
            action_index = action_q_values[category_index].argmax(dim=1).item()
            
        return (
            category_index,
            action_index,
            category_q_values[0, category_index].item(),
            action_q_values[category_index][0, action_index].item()
        )
