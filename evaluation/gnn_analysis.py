"""
GNN Optimization Analysis
======================

This module provides tools for analyzing the performance of GNN-based quantum circuit optimization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from qiskit import QuantumCircuit
import torch
from torch_geometric.utils import to_networkx

from quantum_circuit.dag import CircuitDAG
from agent.gnn_model import GNNEncoder
from environment.hierarchical_action import HierarchicalCircuitAction, ActionCategory
from evaluation.metrics import CircuitMetrics

class GNNOptimizationAnalyzer:
    """
    Analyzer for GNN-based quantum circuit optimization.
    
    This class provides tools for analyzing the performance of GNN-based
    quantum circuit optimization, including attention visualization,
    action distribution analysis, and optimization trajectory analysis.
    """
    
    def __init__(self, figsize=(12, 8), style='default'):
        """
        Initialize the GNN optimization analyzer.
        
        Args:
            figsize (tuple): Figure size for plots.
            style (str): Matplotlib style.
        """
        self.figsize = figsize
        
        # Set plot style
        if style == 'default':
            plt.style.use('default')
        else:
            plt.style.use(style)
        
        # Initialize metrics
        self.metrics = CircuitMetrics()
    
    def visualize_circuit_graph(self, circuit, title=None, save_path=None):
        """
        Visualize the circuit as a graph.
        
        Args:
            circuit: The circuit to visualize.
            title (str): Title for the plot.
            save_path (str): Path to save the plot.
            
        Returns:
            The figure.
        """
        # Convert circuit to DAG
        if isinstance(circuit, QuantumCircuit):
            dag = CircuitDAG(circuit)
        else:
            dag = circuit
        
        # Get node features and edge indices
        node_features, edge_indices, node_labels = dag.get_graph_representation(return_labels=True)
        
        # Convert to networkx graph
        G = nx.DiGraph()
        
        # Add nodes
        for i, label in enumerate(node_labels):
            G.add_node(i, label=label)
        
        # Add edges
        for i in range(edge_indices.shape[1]):
            src, dst = edge_indices[0, i], edge_indices[1, i]
            G.add_edge(src, dst)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', alpha=0.8, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, labels={i: G.nodes[i]['label'] for i in G.nodes}, font_size=10, ax=ax)
        
        # Set title
        if title:
            ax.set_title(title, fontsize=16)
        
        # Remove axis
        ax.axis('off')
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
        
        return fig
    
    def visualize_attention(self, circuit, gnn_encoder, layer_index=0, head_index=0, title=None, save_path=None):
        """
        Visualize the attention weights of a GAT-based GNN encoder.
        
        Args:
            circuit: The circuit to visualize.
            gnn_encoder: The GNN encoder to use.
            layer_index (int): Index of the GAT layer to visualize.
            head_index (int): Index of the attention head to visualize.
            title (str): Title for the plot.
            save_path (str): Path to save the plot.
            
        Returns:
            The figure.
        """
        # Check if the encoder uses GAT
        if not hasattr(gnn_encoder, 'convs') or not hasattr(gnn_encoder.convs[layer_index], 'att'):
            raise ValueError("The GNN encoder must use GAT for attention visualization.")
        
        # Convert circuit to DAG
        if isinstance(circuit, QuantumCircuit):
            dag = CircuitDAG(circuit)
        else:
            dag = circuit
        
        # Get node features and edge indices
        node_features, edge_indices, node_labels = dag.get_graph_representation(return_labels=True)
        
        # Convert to torch tensors
        node_features = torch.FloatTensor(node_features)
        edge_indices = torch.LongTensor(edge_indices)
        
        # Get attention weights
        with torch.no_grad():
            # Forward pass through the encoder
            x = node_features
            for i, conv in enumerate(gnn_encoder.convs):
                if i == layer_index:
                    # Get attention weights for the specified layer
                    _, attention_weights = conv(x, edge_indices, return_attention_weights=True)
                    break
                x = conv(x, edge_indices)
        
        # Extract attention weights for the specified head
        if isinstance(attention_weights, tuple):
            edge_index, att_weights = attention_weights
            # If there are multiple heads, select the specified head
            if att_weights.dim() > 1 and att_weights.size(1) > 1:
                att_weights = att_weights[:, head_index]
        else:
            edge_index, att_weights = edge_indices, attention_weights
        
        # Convert to networkx graph
        G = nx.DiGraph()
        
        # Add nodes
        for i, label in enumerate(node_labels):
            G.add_node(i, label=label)
        
        # Add edges with attention weights
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            weight = att_weights[i].item()
            G.add_edge(src, dst, weight=weight)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', alpha=0.8, ax=ax)
        
        # Draw edges with width proportional to attention weight
        edges = [(u, v) for u, v in G.edges()]
        weights = [G[u][v]['weight'] * 5 for u, v in edges]  # Scale weights for visibility
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, alpha=0.7, edge_color='gray', ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, labels={i: G.nodes[i]['label'] for i in G.nodes}, font_size=10, ax=ax)
        
        # Set title
        if title:
            ax.set_title(title, fontsize=16)
        else:
            ax.set_title(f'Attention Weights (Layer {layer_index}, Head {head_index})', fontsize=16)
        
        # Remove axis
        ax.axis('off')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(weights)/5, vmax=max(weights)/5))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Attention Weight')
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
        
        return fig
    
    def analyze_action_distribution(self, agent, env, num_episodes=10, max_steps=100, title=None, save_path=None):
        """
        Analyze the distribution of actions selected by the agent.
        
        Args:
            agent: The agent to analyze.
            env: The environment to use.
            num_episodes (int): Number of episodes to analyze.
            max_steps (int): Maximum steps per episode.
            title (str): Title for the plot.
            save_path (str): Path to save the plot.
            
        Returns:
            The figure and action distribution data.
        """
        # Initialize action counters
        category_counts = {i: 0 for i in range(env.num_categories)}
        action_counts = {}
        for category in range(env.num_categories):
            action_counts[category] = {i: 0 for i in range(env.num_actions_per_category[category])}
        
        # Run episodes
        for episode in range(num_episodes):
            state, _ = env.reset()
            
            for step in range(max_steps):
                # Select action
                action = agent.select_action(state, deterministic=True)
                
                # Update counters
                if isinstance(action, tuple):
                    category_index, action_index = action
                    category_counts[category_index] += 1
                    action_counts[category_index][action_index] += 1
                elif isinstance(action, HierarchicalCircuitAction):
                    category_index = action.category.value
                    action_index = action.action_index
                    category_counts[category_index] += 1
                    action_counts[category_index][action_index] += 1
                
                # Take step
                next_state, _, done, truncated, _ = env.step(action)
                
                # Move to next state
                state = next_state
                
                # Check if done
                if done or truncated:
                    break
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Plot category distribution
        categories = [ActionCategory(i).name for i in range(env.num_categories)]
        category_values = list(category_counts.values())
        
        ax1.bar(categories, category_values)
        ax1.set_title('Action Category Distribution', fontsize=14)
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot action distribution for the most used category
        most_used_category = max(category_counts, key=category_counts.get)
        most_used_category_name = ActionCategory(most_used_category).name
        
        actions = list(action_counts[most_used_category].keys())
        action_values = list(action_counts[most_used_category].values())
        
        ax2.bar(actions, action_values)
        ax2.set_title(f'Action Distribution for {most_used_category_name}', fontsize=14)
        ax2.set_xlabel('Action Index')
        ax2.set_ylabel('Count')
        
        # Set overall title
        if title:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
        
        # Prepare distribution data
        distribution_data = {
            'category_counts': category_counts,
            'action_counts': action_counts
        }
        
        return fig, distribution_data
    
    def analyze_optimization_trajectory(self, agent, env, metrics_to_track=None, title=None, save_path=None):
        """
        Analyze the trajectory of circuit metrics during optimization.
        
        Args:
            agent: The agent to analyze.
            env: The environment to use.
            metrics_to_track (list): List of metrics to track.
            title (str): Title for the plot.
            save_path (str): Path to save the plot.
            
        Returns:
            The figure and trajectory data.
        """
        if metrics_to_track is None:
            metrics_to_track = ['depth', 'gate_count', 'two_qubit_gate_count', 'cost']
        
        # Initialize trajectory data
        trajectory = {metric: [] for metric in metrics_to_track}
        trajectory['step'] = []
        
        # Reset environment
        state, _ = env.reset()
        
        # Get initial metrics
        initial_metrics = env.state.get_metrics()
        for metric in metrics_to_track:
            if metric in initial_metrics:
                trajectory[metric].append(initial_metrics[metric])
        
        trajectory['step'].append(0)
        
        # Run optimization
        max_steps = env.max_steps
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state, deterministic=True)
            
            # Take step
            next_state, _, done, truncated, info = env.step(action)
            
            # Record metrics
            for metric in metrics_to_track:
                if metric in info:
                    trajectory[metric].append(info[metric])
            
            trajectory['step'].append(step + 1)
            
            # Move to next state
            state = next_state
            
            # Check if done
            if done or truncated:
                break
        
        # Create figure
        fig, axes = plt.subplots(len(metrics_to_track), 1, figsize=(self.figsize[0], self.figsize[1] * len(metrics_to_track) // 2))
        
        if len(metrics_to_track) == 1:
            axes = [axes]
        
        # Plot each metric
        for i, metric in enumerate(metrics_to_track):
            if metric in trajectory:
                axes[i].plot(trajectory['step'], trajectory[metric], marker='o')
                axes[i].set_title(f'{metric.replace("_", " ").title()} vs. Step', fontsize=14)
                axes[i].set_xlabel('Step')
                axes[i].set_ylabel(metric.replace('_', ' ').title())
                axes[i].grid(True, linestyle='--', alpha=0.7)
        
        # Set overall title
        if title:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
        
        return fig, trajectory
    
    def compare_optimization_methods(self, circuits, optimizers, labels=None, metric='depth', title=None, save_path=None):
        """
        Compare different optimization methods on a set of circuits.
        
        Args:
            circuits (list): List of circuits to optimize.
            optimizers (list): List of optimization functions.
            labels (list): List of labels for the optimizers.
            metric (str): Metric to compare.
            title (str): Title for the plot.
            save_path (str): Path to save the plot.
            
        Returns:
            The figure and comparison data.
        """
        if labels is None:
            labels = [f'Optimizer {i+1}' for i in range(len(optimizers))]
        
        # Initialize comparison data
        comparison_data = {
            'circuit': [],
            'original': [],
        }
        
        for label in labels:
            comparison_data[label] = []
        
        # Optimize each circuit with each optimizer
        for i, circuit in enumerate(circuits):
            # Calculate original metric
            original_metrics = self.metrics.calculate_metrics(circuit)
            original_value = original_metrics[metric]
            
            # Add to comparison data
            comparison_data['circuit'].append(f'Circuit {i+1}')
            comparison_data['original'].append(original_value)
            
            # Optimize with each optimizer
            for j, optimizer in enumerate(optimizers):
                # Optimize the circuit
                optimized_circuit = optimizer(circuit)
                
                # Calculate optimized metric
                optimized_metrics = self.metrics.calculate_metrics(optimized_circuit)
                optimized_value = optimized_metrics[metric]
                
                # Add to comparison data
                comparison_data[labels[j]].append(optimized_value)
        
        # Create dataframe
        df = pd.DataFrame(comparison_data)
        
        # Melt dataframe for easier plotting
        df_melted = pd.melt(df, id_vars=['circuit'], value_vars=['original'] + labels, 
                           var_name='optimizer', value_name=metric)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot comparison
        sns.barplot(x='circuit', y=metric, hue='optimizer', data=df_melted, ax=ax)
        
        # Set labels
        ax.set_title(f'Comparison of {metric.replace("_", " ").title()} Reduction', fontsize=16)
        ax.set_xlabel('Circuit')
        ax.set_ylabel(metric.replace('_', ' ').title())
        
        # Set overall title
        if title:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
        
        return fig, comparison_data
    
    def visualize_node_embeddings(self, circuit, gnn_encoder, title=None, save_path=None):
        """
        Visualize the node embeddings of a circuit.
        
        Args:
            circuit: The circuit to visualize.
            gnn_encoder: The GNN encoder to use.
            title (str): Title for the plot.
            save_path (str): Path to save the plot.
            
        Returns:
            The figure.
        """
        # Convert circuit to DAG
        if isinstance(circuit, QuantumCircuit):
            dag = CircuitDAG(circuit)
        else:
            dag = circuit
        
        # Get node features and edge indices
        node_features, edge_indices, node_labels = dag.get_graph_representation(return_labels=True)
        
        # Convert to torch tensors
        node_features = torch.FloatTensor(node_features)
        edge_indices = torch.LongTensor(edge_indices)
        
        # Get node embeddings
        with torch.no_grad():
            node_embeddings = gnn_encoder(node_features, edge_indices).numpy()
        
        # Reduce dimensionality for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        node_embeddings_2d = pca.fit_transform(node_embeddings)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create scatter plot
        scatter = ax.scatter(node_embeddings_2d[:, 0], node_embeddings_2d[:, 1], c=np.arange(len(node_labels)), 
                            cmap='viridis', s=100, alpha=0.8)
        
        # Add labels
        for i, label in enumerate(node_labels):
            ax.annotate(label, (node_embeddings_2d[i, 0], node_embeddings_2d[i, 1]), 
                        fontsize=9, ha='center', va='center')
        
        # Set title
        if title:
            ax.set_title(title, fontsize=16)
        else:
            ax.set_title('Node Embeddings (PCA)', fontsize=16)
        
        # Set labels
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Node Index')
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
        
        return fig
