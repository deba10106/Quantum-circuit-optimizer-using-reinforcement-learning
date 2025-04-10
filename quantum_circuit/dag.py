"""
Circuit DAG Representation
========================

This module provides a directed acyclic graph (DAG) representation of quantum circuits
for use with graph-based algorithms and reinforcement learning.
"""

import numpy as np
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
import torch

class CircuitDAG:
    """
    A directed acyclic graph (DAG) representation of a quantum circuit.
    This representation is suitable for use with graph neural networks.
    """
    
    def __init__(self, circuit=None, dag=None):
        """
        Initialize a circuit DAG.
        
        Args:
            circuit: A Qiskit QuantumCircuit or QuantumCircuit wrapper.
            dag: A Qiskit DAGCircuit.
        """
        if circuit is not None:
            if hasattr(circuit, 'circuit'):
                # If it's our QuantumCircuit wrapper
                self.dag = circuit_to_dag(circuit.circuit)
            else:
                # If it's a Qiskit QuantumCircuit
                self.dag = circuit_to_dag(circuit)
        elif dag is not None:
            self.dag = dag
        else:
            raise ValueError("Either circuit or dag must be provided")
            
        self._graph = None
        self._node_features = None
        self._edge_indices = None
        
    @property
    def graph(self):
        """Get the NetworkX graph representation of the circuit DAG."""
        if self._graph is None:
            self._graph = self._to_networkx()
        return self._graph
    
    def _to_networkx(self):
        """Convert the DAG to a NetworkX graph."""
        import networkx as nx
        
        G = nx.DiGraph()
        
        # Add nodes
        for node in self.dag.op_nodes():
            # In newer Qiskit versions, we need to check the node type differently
            node_attrs = {
                'name': node.name,
                'gate_name': node.op.name,
                'qubits': [q.index for q in node.qargs],
                'params': [float(p) for p in node.op.params] if hasattr(node.op, 'params') else []
            }
            G.add_node(node._node_id, **node_attrs)
            
        # Add edges
        for node in self.dag.topological_op_nodes():
            for successor in self.dag.successors(node):
                # Check if successor is an operation node (not input/output)
                if hasattr(successor, 'op'):
                    G.add_edge(node._node_id, successor._node_id)
                    
        return G
    
    def to_circuit(self):
        """
        Convert the DAG back to a quantum circuit.
        
        Returns:
            A Qiskit QuantumCircuit.
        """
        from quantum_circuit_optimizer.quantum_circuit.circuit import QuantumCircuit
        qiskit_circuit = dag_to_circuit(self.dag)
        return QuantumCircuit(qiskit_circuit)
    
    def get_node_features(self, feature_dim=16):
        """
        Get node features for use with graph neural networks.
        
        Args:
            feature_dim (int): Dimension of the node feature vectors.
            
        Returns:
            torch.Tensor: Node features tensor of shape [num_nodes, feature_dim].
        """
        if self._node_features is not None and self._node_features.shape[1] == feature_dim:
            return self._node_features
            
        # Define gate type encoding
        gate_types = {
            'x': 0, 'y': 1, 'z': 2, 'h': 3, 'cx': 4, 'cz': 5, 'swap': 6,
            'rz': 7, 'rx': 8, 'ry': 9, 'u1': 10, 'u2': 11, 'u3': 12,
            'id': 13, 'barrier': 14, 'measure': 15
        }
        
        # Create node features
        nodes = list(self.graph.nodes(data=True))
        num_nodes = len(nodes)
        features = torch.zeros(num_nodes, feature_dim)
        
        for i, (_, data) in enumerate(nodes):
            # One-hot encoding of gate type
            gate_type = data['gate_name']
            gate_idx = gate_types.get(gate_type, 0)
            if gate_idx < feature_dim:
                features[i, gate_idx] = 1.0
                
            # Number of qubits (normalized)
            if 'qubits' in data and feature_dim > len(gate_types):
                features[i, len(gate_types)] = len(data['qubits']) / 5.0  # Normalize by typical max qubits
                
            # Additional features can be added here
                
        self._node_features = features
        return features
    
    def get_edge_indices(self):
        """
        Get edge indices for use with graph neural networks.
        
        Returns:
            torch.Tensor: Edge indices tensor of shape [2, num_edges].
        """
        if self._edge_indices is not None:
            return self._edge_indices
            
        # Create mapping from node IDs to indices
        nodes = list(self.graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Create edge indices
        edges = list(self.graph.edges())
        num_edges = len(edges)
        edge_indices = torch.zeros(2, num_edges, dtype=torch.long)
        
        for i, (src, dst) in enumerate(edges):
            edge_indices[0, i] = node_to_idx[src]
            edge_indices[1, i] = node_to_idx[dst]
            
        self._edge_indices = edge_indices
        return edge_indices
    
    def get_adjacency_matrix(self):
        """
        Get the adjacency matrix of the graph.
        
        Returns:
            torch.Tensor: Adjacency matrix of shape [num_nodes, num_nodes].
        """
        return torch.tensor(nx.to_numpy_array(self.graph), dtype=torch.float)
    
    def visualize(self, filename=None):
        """
        Visualize the circuit DAG.
        
        Args:
            filename (str, optional): If provided, save the visualization to this file.
                Otherwise, display it.
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        
        # Color nodes by gate type
        gate_colors = {
            'x': 'red', 'y': 'green', 'z': 'blue', 'h': 'purple',
            'cx': 'orange', 'cz': 'pink', 'swap': 'brown',
            'rz': 'cyan', 'rx': 'magenta', 'ry': 'yellow',
            'u1': 'gray', 'u2': 'lightblue', 'u3': 'lightgreen',
            'id': 'white', 'barrier': 'black', 'measure': 'gold'
        }
        
        node_colors = []
        for _, data in self.graph.nodes(data=True):
            gate_type = data['gate_name']
            node_colors.append(gate_colors.get(gate_type, 'gray'))
            
        nx.draw(self.graph, pos, with_labels=True, node_color=node_colors, 
                node_size=500, font_size=10, font_weight='bold')
        
        if filename:
            plt.savefig(filename)
        else:
            plt.show()
            
    def __str__(self):
        """String representation of the circuit DAG."""
        return f"CircuitDAG with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges"
