"""
Robust quantum circuit optimizer that focuses on redundant gate elimination.

This script implements a simplified approach to quantum circuit optimization
that specifically targets redundant gates like H-H pairs.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
import time

from quantum_circuit.dag import CircuitDAG
from evaluation.metrics import CircuitMetrics
from evaluation.visualization import CircuitVisualizer

def create_redundant_gate_circuit(num_qubits=3):
    """Create a circuit with redundant gates (e.g., H-H pairs that cancel out)."""
    qc = QuantumCircuit(num_qubits)
    
    # Add some gates
    for i in range(num_qubits):
        qc.h(i)
    
    # Add redundant H gates (H-H cancels to identity)
    for i in range(num_qubits):
        qc.h(i)
        qc.h(i)
    
    return qc

def find_redundant_gates(circuit_dag):
    """Find redundant gates in the circuit that can be eliminated."""
    redundant_pairs = []
    
    # Get all nodes in the circuit using the graph property
    nodes = list(circuit_dag.graph.nodes(data=True))
    
    # Convert to a list of (node_id, data) tuples for easier processing
    nodes_with_data = [(node_id, data) for node_id, data in nodes]
    
    # Check for adjacent gates of the same type on the same qubit
    for i in range(len(nodes_with_data) - 1):
        node_id1, data1 = nodes_with_data[i]
        
        # Skip if this node is already marked for removal
        if any(node_id1 == pair[0] for pair in redundant_pairs):
            continue
            
        for j in range(i + 1, len(nodes_with_data)):
            node_id2, data2 = nodes_with_data[j]
            
            # Skip if this node is already marked for removal
            if any(node_id2 == pair[1] for pair in redundant_pairs):
                continue
                
            # Check if gates are the same type and operate on the same qubits
            if (data1['gate_name'] == data2['gate_name'] and 
                data1['qubits'] == data2['qubits'] and
                # For now, only consider self-inverse gates like H
                data1['gate_name'] in ['h']):
                
                # Check if there are no gates in between that operate on these qubits
                gates_between = False
                for k in range(i + 1, j):
                    _, data_between = nodes_with_data[k]
                    if any(q in data1['qubits'] for q in data_between['qubits']):
                        gates_between = True
                        break
                
                if not gates_between:
                    redundant_pairs.append((node_id1, node_id2))
    
    return redundant_pairs

def eliminate_redundant_gates(circuit, verbose=True):
    """Eliminate redundant gates from the circuit."""
    # Convert to DAG representation
    circuit_dag = CircuitDAG(circuit)
    
    if verbose:
        print(f"Initial circuit has {len(circuit_dag.graph.nodes())} gates")
    
    # Find redundant gates
    redundant_pairs = find_redundant_gates(circuit_dag)
    
    if verbose:
        print(f"Found {len(redundant_pairs)} redundant gate pairs")
    
    # Create a new circuit without the redundant gates
    optimized_circuit = QuantumCircuit(circuit.num_qubits)
    
    # Get all nodes in the circuit
    nodes = list(circuit_dag.graph.nodes(data=True))
    
    # Collect nodes to remove
    nodes_to_remove = set()
    for pair in redundant_pairs:
        nodes_to_remove.add(pair[0])
        nodes_to_remove.add(pair[1])
    
    # Add gates that are not marked for removal
    for node_id, data in nodes:
        if node_id not in nodes_to_remove:
            # Add the gate to the optimized circuit
            gate_name = data['gate_name']
            qubits = data['qubits']
            
            if gate_name == 'h':
                optimized_circuit.h(qubits[0])
            elif gate_name == 'x':
                optimized_circuit.x(qubits[0])
            elif gate_name == 'cx':
                optimized_circuit.cx(qubits[0], qubits[1])
            elif gate_name == 't':
                optimized_circuit.t(qubits[0])
            elif gate_name == 'tdg':
                optimized_circuit.tdg(qubits[0])
            # Add more gate types as needed
    
    if verbose:
        print(f"Optimized circuit has {optimized_circuit.size()} gates")
    
    return optimized_circuit

def optimize_circuit(circuit, verbose=True):
    """Apply multiple optimization passes to the circuit."""
    if verbose:
        print("Original circuit:")
        print(circuit)
    
    # Apply optimization
    start_time = time.time()
    optimized_circuit = eliminate_redundant_gates(circuit, verbose)
    optimization_time = time.time() - start_time
    
    if verbose:
        print(f"\nOptimization completed in {optimization_time:.4f} seconds")
        print("\nOptimized circuit:")
        print(optimized_circuit)
    
    # Calculate metrics
    metrics = CircuitMetrics()
    original_metrics = metrics.calculate_metrics(circuit)
    optimized_metrics = metrics.calculate_metrics(optimized_circuit, circuit)
    
    if verbose:
        print("\nMetrics comparison:")
        print(f"Original depth: {original_metrics['depth']}")
        print(f"Optimized depth: {optimized_metrics['depth']}")
        print(f"Depth reduction: {optimized_metrics['depth_reduction']:.2f}%")
        
        print(f"Original gate count: {original_metrics['gate_count']}")
        print(f"Optimized gate count: {optimized_metrics['gate_count']}")
        print(f"Gate count reduction: {optimized_metrics['gate_count_reduction']:.2f}%")
    
    return optimized_circuit, original_metrics, optimized_metrics

def run_tests():
    """Run tests on various circuits."""
    # Test on redundant gate circuit
    print("="*50)
    print("Testing on redundant gate circuit")
    print("="*50)
    
    circuit = create_redundant_gate_circuit(num_qubits=3)
    optimized_circuit, original_metrics, optimized_metrics = optimize_circuit(circuit)
    
    # Visualize the circuits
    try:
        visualizer = CircuitVisualizer()
        fig = visualizer.plot_circuit_comparison(circuit, optimized_circuit, 
                                              title="Optimization for Redundant Gate Circuit")
        plt.savefig("robust_optimizer_circuit_comparison.png")
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
    
    return {
        "original_metrics": original_metrics,
        "optimized_metrics": optimized_metrics,
        "original_circuit": circuit,
        "optimized_circuit": optimized_circuit
    }

if __name__ == "__main__":
    run_tests()
