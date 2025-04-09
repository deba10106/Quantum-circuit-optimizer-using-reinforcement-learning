"""
Direct Optimization Script
========================

This script directly runs the quantum circuit optimization without relying on the package structure.
"""

import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.visualization import plot_histogram
from qiskit import Aer, execute

# Import the cost database
from cost_database.cost_database import CostDatabase

def optimize_circuit():
    """Run a simple circuit optimization."""
    # Create a cost database
    cost_database = CostDatabase()
    print("Created cost database successfully")
    
    # Create a quantum circuit (QFT on 3 qubits)
    print("Creating a QFT circuit with 3 qubits...")
    circuit = QFT(3)
    
    # Print the original circuit
    print("Original circuit:")
    print(circuit)
    
    # Get the circuit depth
    depth = circuit.depth()
    print(f"Original circuit depth: {depth}")
    
    # Get the gate counts
    gate_counts = circuit.count_ops()
    print(f"Original gate counts: {gate_counts}")
    
    # Transpile the circuit to optimize it
    from qiskit import transpile
    optimized_circuit = transpile(circuit, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=3)
    
    # Print the optimized circuit
    print("\nOptimized circuit:")
    print(optimized_circuit)
    
    # Get the optimized circuit depth
    opt_depth = optimized_circuit.depth()
    print(f"Optimized circuit depth: {opt_depth}")
    
    # Get the optimized gate counts
    opt_gate_counts = optimized_circuit.count_ops()
    print(f"Optimized gate counts: {opt_gate_counts}")
    
    # Calculate improvement
    depth_improvement = (depth - opt_depth) / depth * 100
    print(f"Depth improvement: {depth_improvement:.2f}%")
    
    # Visualize the circuits
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    circuit.draw('mpl', ax=ax1)
    ax1.set_title('Original Circuit')
    optimized_circuit.draw('mpl', ax=ax2)
    ax2.set_title('Optimized Circuit')
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig('circuit_optimization.png')
    print("Visualization saved to 'circuit_optimization.png'")
    
    return circuit, optimized_circuit

if __name__ == '__main__':
    optimize_circuit()
