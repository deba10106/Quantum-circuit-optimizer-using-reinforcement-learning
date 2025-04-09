"""
Simple Quantum Circuit Optimizer
==============================

This script demonstrates quantum circuit optimization using Qiskit's transpiler
without relying on the complex package structure.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile, Aer
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QFT
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller, Optimize1qGates, CommutativeCancellation

def create_random_circuit(num_qubits, depth, seed=None):
    """Create a random quantum circuit."""
    from qiskit.circuit.random import random_circuit
    return random_circuit(num_qubits, depth, max_operands=2, measure=False, seed=seed)

def optimize_circuit(circuit, optimization_level=3):
    """Optimize a quantum circuit using Qiskit's transpiler."""
    # Define the basis gates
    basis_gates = ['u1', 'u2', 'u3', 'cx']
    
    # Transpile the circuit
    optimized_circuit = transpile(
        circuit, 
        basis_gates=basis_gates,
        optimization_level=optimization_level
    )
    
    return optimized_circuit

def calculate_metrics(circuit):
    """Calculate metrics for a quantum circuit."""
    # Calculate depth
    depth = circuit.depth()
    
    # Calculate gate counts
    gate_counts = circuit.count_ops()
    
    # Calculate total number of gates
    total_gates = sum(gate_counts.values())
    
    # Calculate number of two-qubit gates
    two_qubit_gates = sum(count for gate, count in gate_counts.items() 
                         if gate in ['cx', 'cz', 'swap'])
    
    return {
        'depth': depth,
        'gate_counts': gate_counts,
        'total_gates': total_gates,
        'two_qubit_gates': two_qubit_gates
    }

def visualize_optimization(original_circuit, optimized_circuit):
    """Visualize the original and optimized circuits."""
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Draw the original circuit
    original_circuit.draw('mpl', ax=ax1)
    ax1.set_title('Original Circuit')
    
    # Draw the optimized circuit
    optimized_circuit.draw('mpl', ax=ax2)
    ax2.set_title('Optimized Circuit')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('circuit_optimization.png')
    print("Visualization saved to 'circuit_optimization.png'")
    
    return fig

def run_simulation(circuit, shots=1024):
    """Run a simulation of the circuit."""
    # Get the simulator
    simulator = Aer.get_backend('qasm_simulator')
    
    # Run the simulation
    result = simulator.run(circuit, shots=shots).result()
    
    # Get the counts
    counts = result.get_counts(circuit)
    
    return counts

def main():
    """Main function to demonstrate quantum circuit optimization."""
    # Create a quantum circuit (QFT on 4 qubits)
    print("Creating a QFT circuit with 4 qubits...")
    circuit = QFT(4)
    
    # Print the original circuit
    print("Original circuit:")
    print(circuit)
    
    # Calculate metrics for the original circuit
    original_metrics = calculate_metrics(circuit)
    print(f"Original circuit metrics:")
    print(f"  Depth: {original_metrics['depth']}")
    print(f"  Gate counts: {original_metrics['gate_counts']}")
    print(f"  Total gates: {original_metrics['total_gates']}")
    print(f"  Two-qubit gates: {original_metrics['two_qubit_gates']}")
    
    # Optimize the circuit
    print("\nOptimizing the circuit...")
    optimized_circuit = optimize_circuit(circuit)
    
    # Print the optimized circuit
    print("Optimized circuit:")
    print(optimized_circuit)
    
    # Calculate metrics for the optimized circuit
    optimized_metrics = calculate_metrics(optimized_circuit)
    print(f"Optimized circuit metrics:")
    print(f"  Depth: {optimized_metrics['depth']}")
    print(f"  Gate counts: {optimized_metrics['gate_counts']}")
    print(f"  Total gates: {optimized_metrics['total_gates']}")
    print(f"  Two-qubit gates: {optimized_metrics['two_qubit_gates']}")
    
    # Calculate improvement
    depth_improvement = ((original_metrics['depth'] - optimized_metrics['depth']) / 
                         original_metrics['depth'] * 100)
    gate_improvement = ((original_metrics['total_gates'] - optimized_metrics['total_gates']) / 
                        original_metrics['total_gates'] * 100)
    two_qubit_improvement = ((original_metrics['two_qubit_gates'] - optimized_metrics['two_qubit_gates']) / 
                            max(1, original_metrics['two_qubit_gates']) * 100)
    
    print(f"\nImprovements:")
    print(f"  Depth: {depth_improvement:.2f}%")
    print(f"  Total gates: {gate_improvement:.2f}%")
    print(f"  Two-qubit gates: {two_qubit_improvement:.2f}%")
    
    # Visualize the optimization
    print("\nVisualizing the optimization...")
    visualize_optimization(circuit, optimized_circuit)
    
    # Try with a random circuit
    print("\nCreating a random circuit with 4 qubits and depth 10...")
    random_circuit = create_random_circuit(4, 10, seed=42)
    
    # Print the random circuit
    print("Random circuit:")
    print(random_circuit)
    
    # Calculate metrics for the random circuit
    random_metrics = calculate_metrics(random_circuit)
    print(f"Random circuit metrics:")
    print(f"  Depth: {random_metrics['depth']}")
    print(f"  Gate counts: {random_metrics['gate_counts']}")
    print(f"  Total gates: {random_metrics['total_gates']}")
    print(f"  Two-qubit gates: {random_metrics['two_qubit_gates']}")
    
    # Optimize the random circuit
    print("\nOptimizing the random circuit...")
    optimized_random = optimize_circuit(random_circuit)
    
    # Print the optimized random circuit
    print("Optimized random circuit:")
    print(optimized_random)
    
    # Calculate metrics for the optimized random circuit
    optimized_random_metrics = calculate_metrics(optimized_random)
    print(f"Optimized random circuit metrics:")
    print(f"  Depth: {optimized_random_metrics['depth']}")
    print(f"  Gate counts: {optimized_random_metrics['gate_counts']}")
    print(f"  Total gates: {optimized_random_metrics['total_gates']}")
    print(f"  Two-qubit gates: {optimized_random_metrics['two_qubit_gates']}")
    
    # Calculate improvement for the random circuit
    random_depth_improvement = ((random_metrics['depth'] - optimized_random_metrics['depth']) / 
                               random_metrics['depth'] * 100)
    random_gate_improvement = ((random_metrics['total_gates'] - optimized_random_metrics['total_gates']) / 
                              random_metrics['total_gates'] * 100)
    random_two_qubit_improvement = ((random_metrics['two_qubit_gates'] - optimized_random_metrics['two_qubit_gates']) / 
                                   max(1, random_metrics['two_qubit_gates']) * 100)
    
    print(f"\nRandom circuit improvements:")
    print(f"  Depth: {random_depth_improvement:.2f}%")
    print(f"  Total gates: {random_gate_improvement:.2f}%")
    print(f"  Two-qubit gates: {random_two_qubit_improvement:.2f}%")
    
    # Visualize the random circuit optimization
    print("\nVisualizing the random circuit optimization...")
    fig = visualize_optimization(random_circuit, optimized_random)
    plt.savefig('random_circuit_optimization.png')
    print("Visualization saved to 'random_circuit_optimization.png'")

if __name__ == "__main__":
    main()
