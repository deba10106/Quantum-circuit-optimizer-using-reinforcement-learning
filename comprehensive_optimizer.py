"""
Comprehensive Quantum Circuit Optimizer

This script implements a comprehensive approach to quantum circuit optimization
that targets multiple optimization patterns:
1. Redundant gate elimination (e.g., H-H pairs)
2. Gate commutation and reordering
3. Simple gate decomposition
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
import time
import copy

from quantum_circuit.dag import CircuitDAG
from evaluation.metrics import CircuitMetrics
from evaluation.visualization import CircuitVisualizer

# Dictionary of self-inverse gates (gates that cancel themselves when applied twice)
SELF_INVERSE_GATES = {'h', 'x', 'y', 'z', 'cx', 'cy', 'cz', 'swap'}

# Dictionary of gate equivalences (gates that can be replaced with simpler sequences)
GATE_EQUIVALENCES = {
    # Add more equivalences as needed
}

def create_test_circuits():
    """Create a suite of test circuits with known optimization opportunities."""
    circuits = {}
    
    # 1. Redundant gate circuit
    qc1 = QuantumCircuit(3)
    # Add some gates
    for i in range(3):
        qc1.h(i)
    # Add redundant H gates (H-H cancels to identity)
    for i in range(3):
        qc1.h(i)
        qc1.h(i)
    circuits["redundant_gates"] = qc1
    
    # 2. Commutable gate circuit
    qc2 = QuantumCircuit(3)
    # Add initial layer
    for i in range(3):
        qc2.h(i)
    # Add CX gates in a pattern that can be optimized by reordering
    qc2.cx(0, 1)
    qc2.cx(1, 2)
    qc2.cx(0, 2)  # This can be reordered to reduce depth
    # Add single-qubit gates that can be commuted
    qc2.t(0)
    qc2.t(1)
    qc2.t(2)
    # Add more CX gates
    qc2.cx(0, 1)
    qc2.cx(1, 2)
    circuits["commutable_gates"] = qc2
    
    # 3. Decomposable gate circuit
    qc3 = QuantumCircuit(3)
    # Add some gates
    for i in range(3):
        qc3.h(i)
    # Add SWAP gates (which can be decomposed into 3 CNOT gates)
    for i in range(2):
        qc3.swap(i, i+1)
    # Add some more gates
    for i in range(3):
        qc3.t(i)
    circuits["decomposable_gates"] = qc3
    
    # 4. QFT circuit
    qc4 = QFT(3)
    circuits["qft"] = qc4
    
    # 5. Mixed circuit with multiple optimization opportunities
    qc5 = QuantumCircuit(3)
    # Add some gates
    for i in range(3):
        qc5.h(i)
    # Add redundant gates
    qc5.x(0)
    qc5.x(0)
    # Add some entangling gates
    qc5.cx(0, 1)
    qc5.cx(1, 2)
    # Add more redundant gates
    qc5.h(0)
    qc5.h(0)
    # Add a swap
    qc5.swap(0, 2)
    # Add some phase gates
    qc5.t(0)
    qc5.t(1)
    qc5.t(2)
    circuits["mixed_circuit"] = qc5
    
    return circuits

def find_redundant_gates(circuit_dag):
    """Find redundant gates in the circuit that can be eliminated."""
    redundant_pairs = []
    
    # Get all nodes in the circuit using the graph property
    nodes = list(circuit_dag.graph.nodes(data=True))
    
    # Convert to a list of (node_id, data) tuples for easier processing
    nodes_with_data = [(node_id, data) for node_id, data in nodes]
    
    # Check for adjacent gates of the same type on the same qubits
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
                # Only consider self-inverse gates
                data1['gate_name'] in SELF_INVERSE_GATES):
                
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
            elif gate_name == 'y':
                optimized_circuit.y(qubits[0])
            elif gate_name == 'z':
                optimized_circuit.z(qubits[0])
            elif gate_name == 'cx':
                optimized_circuit.cx(qubits[0], qubits[1])
            elif gate_name == 'cy':
                optimized_circuit.cy(qubits[0], qubits[1])
            elif gate_name == 'cz':
                optimized_circuit.cz(qubits[0], qubits[1])
            elif gate_name == 'swap':
                optimized_circuit.swap(qubits[0], qubits[1])
            elif gate_name == 't':
                optimized_circuit.t(qubits[0])
            elif gate_name == 'tdg':
                optimized_circuit.tdg(qubits[0])
            elif gate_name == 's':
                optimized_circuit.s(qubits[0])
            elif gate_name == 'sdg':
                optimized_circuit.sdg(qubits[0])
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
    
    # First pass: eliminate redundant gates
    optimized_circuit = eliminate_redundant_gates(circuit, verbose)
    
    # Additional optimization passes can be added here
    
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

def run_comprehensive_tests(save_dir="comprehensive_results"):
    """Run tests on various circuits."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Create test circuits
    circuits = create_test_circuits()
    
    # Dictionary to store all results
    all_results = {}
    
    # Test each circuit
    for circuit_name, circuit in circuits.items():
        print("="*50)
        print(f"Testing on {circuit_name} circuit")
        print("="*50)
        
        try:
            optimized_circuit, original_metrics, optimized_metrics = optimize_circuit(circuit)
            
            # Visualize the circuits
            try:
                visualizer = CircuitVisualizer()
                fig = visualizer.plot_circuit_comparison(circuit, optimized_circuit, 
                                                      title=f"Optimization for {circuit_name}")
                plt.savefig(f"{save_dir}/{circuit_name}_comparison.png")
            except Exception as e:
                print(f"Error during visualization: {str(e)}")
            
            # Store results
            all_results[circuit_name] = {
                "original_metrics": original_metrics,
                "optimized_metrics": optimized_metrics,
                "depth_reduction": optimized_metrics['depth_reduction'],
                "gate_count_reduction": optimized_metrics['gate_count_reduction']
            }
            
        except Exception as e:
            print(f"Error optimizing {circuit_name} circuit: {str(e)}")
            all_results[circuit_name] = {
                "depth_reduction": 0,
                "gate_count_reduction": 0
            }
    
    # Generate summary report
    print("\n" + "="*50)
    print("SUMMARY OF RESULTS")
    print("="*50)
    
    print("\nDepth Reduction:")
    for circuit_name, results in all_results.items():
        print(f"{circuit_name}: {results.get('depth_reduction', 0):.2f}%")
    
    print("\nGate Count Reduction:")
    for circuit_name, results in all_results.items():
        print(f"{circuit_name}: {results.get('gate_count_reduction', 0):.2f}%")
    
    # Create summary plots
    try:
        plt.figure(figsize=(12, 6))
        
        # Depth reduction plot
        plt.subplot(1, 2, 1)
        circuit_names = list(all_results.keys())
        depth_reductions = [all_results[name].get('depth_reduction', 0) for name in circuit_names]
        plt.bar(circuit_names, depth_reductions)
        plt.title('Depth Reduction by Circuit Type')
        plt.ylabel('Reduction (%)')
        plt.xticks(rotation=45)
        
        # Gate count reduction plot
        plt.subplot(1, 2, 2)
        gate_reductions = [all_results[name].get('gate_count_reduction', 0) for name in circuit_names]
        plt.bar(circuit_names, gate_reductions)
        plt.title('Gate Count Reduction by Circuit Type')
        plt.ylabel('Reduction (%)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/summary_results.png")
    except Exception as e:
        print(f"Error creating summary plots: {str(e)}")
    
    return all_results

if __name__ == "__main__":
    run_comprehensive_tests()
