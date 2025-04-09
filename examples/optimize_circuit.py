"""
Example: Optimize a Quantum Circuit
=================================

This example demonstrates how to use the quantum circuit optimizer to optimize a quantum circuit.
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

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the required modules
from quantum_circuit_optimizer.quantum_circuit.circuit import QuantumCircuit as QCO
from quantum_circuit_optimizer.cost_database.cost_database import CostDatabase
from quantum_circuit_optimizer.environment.quantum_environment import QuantumEnvironment
from quantum_circuit_optimizer.agent.dqn_agent import DQNAgent
from quantum_circuit_optimizer.evaluation.metrics import CircuitMetrics
from quantum_circuit_optimizer.evaluation.visualization import CircuitVisualizer

def main():
    """Main function."""
    # Create a cost database
    cost_database = CostDatabase()
    
    # Create a quantum circuit (QFT on 5 qubits)
    print("Creating a QFT circuit with 5 qubits...")
    circuit = QFT(5)
    
    # Print the original circuit
    print("Original circuit:")
    print(circuit)
    
    # Create an environment
    print("Creating the RL environment...")
    env = QuantumEnvironment(
        initial_circuit=circuit,
        cost_database=cost_database,
        max_steps=50,
        depth_weight=0.3,
        cost_weight=0.3,
        error_weight=0.4,
        feature_dim=32,
        render_mode=None
    )
    
    # Create a DQN agent
    print("Creating the DQN agent...")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update=10,
        device='cpu'  # Force CPU usage instead of auto-detection
    )
    
    # Train the agent (for a small number of steps as an example)
    print("Training the agent for 1000 steps...")
    train_results = agent.train(env, 1000)
    
    # Optimize the circuit
    print("Optimizing the circuit...")
    optimized_circuit, optimization_info = agent.optimize_circuit(
        env, circuit, max_steps=50, render=False
    )
    
    # Print optimization results
    print("Optimization results:")
    print(f"  Original depth: {optimization_info['initial_depth']}")
    print(f"  Optimized depth: {optimization_info['optimized_depth']}")
    print(f"  Depth reduction: {optimization_info['depth_reduction']} "
          f"({optimization_info['depth_reduction'] / optimization_info['initial_depth'] * 100:.2f}%)")
    
    print(f"  Original gate count: {optimization_info['initial_gate_count']}")
    print(f"  Optimized gate count: {optimization_info['optimized_gate_count']}")
    print(f"  Gate count reduction: {optimization_info['gate_count_reduction']} "
          f"({optimization_info['gate_count_reduction'] / optimization_info['initial_gate_count'] * 100:.2f}%)")
    
    if optimization_info['cost_reduction'] is not None:
        print(f"  Original cost: {optimization_info['initial_cost']:.4f}")
        print(f"  Optimized cost: {optimization_info['optimized_cost']:.4f}")
        print(f"  Cost reduction: {optimization_info['cost_reduction']:.4f} "
              f"({optimization_info['cost_reduction'] / optimization_info['initial_cost'] * 100:.2f}%)")
    
    if optimization_info['error_reduction'] is not None:
        print(f"  Original error: {optimization_info['initial_error']:.4f}")
        print(f"  Optimized error: {optimization_info['optimized_error']:.4f}")
        print(f"  Error reduction: {optimization_info['error_reduction']:.4f} "
              f"({optimization_info['error_reduction'] / optimization_info['initial_error'] * 100:.2f}%)")
    
    # Check circuit equivalence
    metrics = CircuitMetrics(cost_database)
    equivalent = metrics.check_equivalence(circuit, optimized_circuit.circuit)
    print(f"Circuits are equivalent: {equivalent}")
    
    # Print the optimized circuit
    print("Optimized circuit:")
    print(optimized_circuit.circuit)
    
    # Visualize results
    visualizer = CircuitVisualizer()
    
    # Plot circuit comparison
    plt_comparison = visualizer.plot_circuit_comparison(
        circuit, optimized_circuit.circuit,
        title='Circuit Comparison'
    )
    
    # Plot optimization metrics
    metrics_dict = metrics.calculate_metrics(optimized_circuit.circuit, circuit)
    plt_metrics = visualizer.plot_optimization_metrics(
        metrics_dict,
        title='Optimization Metrics'
    )
    
    # Show the plots
    plt.show()

if __name__ == '__main__':
    main()
