"""
Circuit State
===========

This module provides the state representation for the quantum circuit optimization environment.
"""

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

class CircuitState:
    """
    Represents the state of a quantum circuit in the optimization environment.
    This includes the circuit itself and various features that describe it.
    """
    
    def __init__(self, circuit, cost_database=None):
        """
        Initialize a circuit state.
        
        Args:
            circuit: A Qiskit QuantumCircuit or our QuantumCircuit wrapper.
            cost_database: A CostDatabase instance for calculating costs and errors.
        """
        # Store the circuit
        if hasattr(circuit, 'circuit'):
            # If it's our QuantumCircuit wrapper
            self.circuit = circuit
            self.qiskit_circuit = circuit.circuit
        else:
            # If it's a Qiskit QuantumCircuit
            from quantum_circuit.circuit import QuantumCircuit as QCO
            self.circuit = QCO(circuit)
            self.qiskit_circuit = circuit
            
        # Store the cost database
        self.cost_database = cost_database
        
        # Cache for computed properties
        self._depth = None
        self._gate_count = None
        self._two_qubit_gate_count = None
        self._gate_type_counts = None
        self._cost = None
        self._error = None
        self._feature_vector = None
        
    @property
    def depth(self):
        """Get the circuit depth."""
        if self._depth is None:
            self._depth = self.qiskit_circuit.depth()
        return self._depth
    
    @property
    def gate_count(self):
        """Get the total gate count."""
        if self._gate_count is None:
            self._gate_count = sum(1 for op in self.qiskit_circuit.data if op[0].name not in ['barrier', 'snapshot'])
        return self._gate_count
    
    @property
    def two_qubit_gate_count(self):
        """Get the count of two-qubit gates."""
        if self._two_qubit_gate_count is None:
            self._two_qubit_gate_count = sum(1 for op in self.qiskit_circuit.data 
                                           if len(op[1]) > 1 and op[0].name not in ['barrier', 'snapshot'])
        return self._two_qubit_gate_count
    
    @property
    def gate_type_counts(self):
        """Get a dictionary of gate types and their counts."""
        if self._gate_type_counts is None:
            self._gate_type_counts = {}
            for op in self.qiskit_circuit.data:
                gate_name = op[0].name
                if gate_name not in ['barrier', 'snapshot']:
                    self._gate_type_counts[gate_name] = self._gate_type_counts.get(gate_name, 0) + 1
        return self._gate_type_counts
    
    @property
    def cost(self):
        """Get the total cost of the circuit."""
        if self._cost is None and self.cost_database is not None:
            self._cost = self.cost_database.calculate_circuit_cost(self.qiskit_circuit)
        return self._cost
    
    @property
    def error(self):
        """Get the estimated error rate of the circuit."""
        if self._error is None and self.cost_database is not None:
            self._error = self.cost_database.calculate_circuit_error(self.qiskit_circuit)
        return self._error
    
    def get_feature_vector(self, feature_dim=None):
        """
        Get a feature vector representing the circuit state.
        
        Args:
            feature_dim (int, optional): Dimension of the feature vector.
                If None, uses a variable-length vector based on the circuit properties.
                
        Returns:
            np.ndarray: A feature vector representing the circuit state.
        """
        if self._feature_vector is not None and (feature_dim is None or len(self._feature_vector) == feature_dim):
            return self._feature_vector
            
        # Basic circuit features
        features = [
            self.depth,
            self.gate_count,
            self.two_qubit_gate_count,
            self.qiskit_circuit.num_qubits
        ]
        
        # Add gate type counts
        gate_types = self.gate_type_counts
        common_gates = ['x', 'y', 'z', 'h', 'cx', 'cz', 'swap', 'rz', 'rx', 'ry', 'u1', 'u2', 'u3', 'ecr']
        for gate in common_gates:
            features.append(gate_types.get(gate, 0))
            
        # Add cost and error if available
        if self.cost is not None:
            features.append(self.cost)
        if self.error is not None:
            features.append(self.error)
            
        # Convert to numpy array
        features = np.array(features, dtype=np.float32)
        
        # Pad or truncate to feature_dim if specified
        if feature_dim is not None:
            if len(features) < feature_dim:
                # Pad with zeros
                features = np.pad(features, (0, feature_dim - len(features)))
            elif len(features) > feature_dim:
                # Truncate
                features = features[:feature_dim]
                
        self._feature_vector = features
        return features
    
    def get_graph_representation(self):
        """
        Get a graph representation of the circuit for use with graph neural networks.
        
        Returns:
            tuple: (node_features, edge_indices, adjacency_matrix)
        """
        from quantum_circuit.dag import CircuitDAG
        
        # Create a CircuitDAG from the circuit
        dag = CircuitDAG(self.qiskit_circuit)
        
        # Get node features and edge indices
        node_features = dag.get_node_features()
        edge_indices = dag.get_edge_indices()
        adjacency_matrix = dag.get_adjacency_matrix()
        
        return node_features, edge_indices, adjacency_matrix
    
    def is_equivalent_to(self, other_state, epsilon=1e-10):
        """
        Check if this circuit state is equivalent to another circuit state.
        
        Args:
            other_state (CircuitState): The other circuit state to compare with.
            epsilon (float): Tolerance for equivalence check.
            
        Returns:
            bool: True if the circuits are equivalent, False otherwise.
        """
        try:
            from qiskit.quantum_info import Operator
            
            # Create operators for both circuits
            op1 = Operator(self.qiskit_circuit)
            op2 = Operator(other_state.qiskit_circuit)
            
            # Check if they're equivalent
            return op1.equiv(op2, epsilon)
        except Exception as e:
            # If there's an error (e.g., different dimensions), they're not equivalent
            print(f"Error in equivalence check: {e}")
            return False
    
    def copy(self):
        """
        Create a copy of the circuit state.
        
        Returns:
            CircuitState: A copy of the circuit state.
        """
        return CircuitState(self.qiskit_circuit.copy(), self.cost_database)
    
    def __str__(self):
        """String representation of the circuit state."""
        return (f"CircuitState: depth={self.depth}, gates={self.gate_count}, "
                f"two-qubit gates={self.two_qubit_gate_count}, "
                f"cost={self.cost}, error={self.error}")
    
    def __repr__(self):
        """Representation of the circuit state."""
        return str(self)
