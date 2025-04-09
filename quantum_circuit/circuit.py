"""
Quantum Circuit Representation
============================

This module provides a wrapper around Qiskit's QuantumCircuit with additional
functionality for optimization.
"""

import numpy as np
from qiskit import QuantumCircuit as QiskitCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller, Optimize1qGates, CommutativeCancellation
import networkx as nx

class QuantumCircuit:
    """
    A wrapper around Qiskit's QuantumCircuit with additional functionality
    for optimization and analysis.
    """
    
    def __init__(self, circuit=None, num_qubits=None):
        """
        Initialize a quantum circuit.
        
        Args:
            circuit (QiskitCircuit, optional): A Qiskit circuit to wrap.
            num_qubits (int, optional): Number of qubits to initialize a new circuit.
        """
        if circuit is not None:
            self.circuit = circuit
        elif num_qubits is not None:
            self.circuit = QiskitCircuit(num_qubits)
        else:
            raise ValueError("Either circuit or num_qubits must be provided")
            
        # Cache for computed properties
        self._depth = None
        self._gate_count = None
        self._two_qubit_gate_count = None
        self._dag = None
        
    @property
    def depth(self):
        """Get the circuit depth."""
        if self._depth is None:
            self._depth = self.circuit.depth()
        return self._depth
    
    @property
    def gate_count(self):
        """Get the total gate count."""
        if self._gate_count is None:
            self._gate_count = sum(1 for op in self.circuit.data if op[0].name not in ['barrier', 'snapshot'])
        return self._gate_count
    
    @property
    def two_qubit_gate_count(self):
        """Get the count of two-qubit gates."""
        if self._two_qubit_gate_count is None:
            self._two_qubit_gate_count = sum(1 for op in self.circuit.data 
                                           if len(op[1]) > 1 and op[0].name not in ['barrier', 'snapshot'])
        return self._two_qubit_gate_count
    
    @property
    def dag(self):
        """Get the directed acyclic graph representation of the circuit."""
        if self._dag is None:
            self._dag = circuit_to_dag(self.circuit)
        return self._dag
    
    def to_networkx(self):
        """Convert the circuit DAG to a NetworkX graph for use with GNNs."""
        G = nx.DiGraph()
        
        # Add nodes
        for node in self.dag.topological_op_nodes():
            node_attrs = {
                'gate_type': node.name,
                'num_qubits': len(node.qargs),
                'position': node.op.position if hasattr(node.op, 'position') else None
            }
            G.add_node(node._node_id, **node_attrs)
            
        # Add edges
        for node in self.dag.topological_op_nodes():
            for successor in self.dag.successors(node):
                if successor.type == 'op':
                    G.add_edge(node._node_id, successor._node_id)
                    
        return G
    
    def apply_transformation(self, transformation):
        """
        Apply a circuit transformation.
        
        Args:
            transformation: A function that takes a circuit and returns a transformed circuit.
            
        Returns:
            QuantumCircuit: A new QuantumCircuit with the transformation applied.
        """
        new_circuit = transformation(self.circuit)
        return QuantumCircuit(new_circuit)
    
    def decompose_to_basis_gates(self, basis_gates=None):
        """
        Decompose the circuit to a specific basis gate set.
        
        Args:
            basis_gates (list, optional): List of basis gates to decompose to.
                If None, uses ['u1', 'u2', 'u3', 'cx'].
                
        Returns:
            QuantumCircuit: A new QuantumCircuit decomposed to the basis gates.
        """
        if basis_gates is None:
            basis_gates = ['u1', 'u2', 'u3', 'cx']
            
        pm = PassManager(Unroller(basis_gates))
        new_circuit = pm.run(self.circuit)
        return QuantumCircuit(new_circuit)
    
    def optimize_1q_gates(self):
        """
        Optimize single-qubit gates in the circuit.
        
        Returns:
            QuantumCircuit: A new QuantumCircuit with optimized single-qubit gates.
        """
        pm = PassManager(Optimize1qGates())
        new_circuit = pm.run(self.circuit)
        return QuantumCircuit(new_circuit)
    
    def cancel_commutative_gates(self):
        """
        Cancel commutative gates in the circuit.
        
        Returns:
            QuantumCircuit: A new QuantumCircuit with canceled commutative gates.
        """
        pm = PassManager(CommutativeCancellation())
        new_circuit = pm.run(self.circuit)
        return QuantumCircuit(new_circuit)
    
    def get_gate_types(self):
        """
        Get a dictionary of gate types and their counts.
        
        Returns:
            dict: A dictionary mapping gate types to their counts.
        """
        gate_types = {}
        for op in self.circuit.data:
            gate_name = op[0].name
            if gate_name not in ['barrier', 'snapshot']:
                gate_types[gate_name] = gate_types.get(gate_name, 0) + 1
        return gate_types
    
    def get_circuit_features(self):
        """
        Get a feature vector representing the circuit.
        
        Returns:
            np.ndarray: A feature vector representing the circuit.
        """
        features = [
            self.depth,
            self.gate_count,
            self.two_qubit_gate_count,
            self.circuit.num_qubits
        ]
        
        # Add gate type counts
        gate_types = self.get_gate_types()
        common_gates = ['x', 'y', 'z', 'h', 'cx', 'cz', 'swap', 'rz', 'rx', 'ry', 'u1', 'u2', 'u3']
        for gate in common_gates:
            features.append(gate_types.get(gate, 0))
            
        return np.array(features)
    
    def to_qasm(self):
        """
        Convert the circuit to QASM format.
        
        Returns:
            str: The QASM representation of the circuit.
        """
        return self.circuit.qasm()
    
    def from_qasm(self, qasm_str):
        """
        Create a circuit from a QASM string.
        
        Args:
            qasm_str (str): The QASM string to convert.
            
        Returns:
            QuantumCircuit: A new QuantumCircuit from the QASM string.
        """
        circuit = QiskitCircuit.from_qasm_str(qasm_str)
        return QuantumCircuit(circuit)
    
    def copy(self):
        """
        Create a copy of the circuit.
        
        Returns:
            QuantumCircuit: A copy of the circuit.
        """
        return QuantumCircuit(self.circuit.copy())
    
    def __str__(self):
        """String representation of the circuit."""
        return str(self.circuit)
    
    def __repr__(self):
        """Representation of the circuit."""
        return repr(self.circuit)
