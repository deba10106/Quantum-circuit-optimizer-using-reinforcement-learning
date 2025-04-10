"""
Test script for the GNN-based quantum circuit optimizer.

This script creates a simple quantum circuit and tests the GNN-based optimizer
with both standard and hierarchical action spaces.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT

from quantum_circuit.dag import CircuitDAG
from environment.gnn_quantum_environment import GNNQuantumEnvironment
from agent.gnn_dqn_agent import GNNDQNAgent
from agent.hierarchical_gnn_dqn_agent import HierarchicalGNNDQNAgent
from cost_database.cost_database import CostDatabase
from evaluation.metrics import CircuitMetrics
from evaluation.visualization import CircuitVisualizer

def create_test_circuit(num_qubits=3):
    """Create a simple test circuit."""
    qc = QuantumCircuit(num_qubits)
    
    # Add some gates
    for i in range(num_qubits):
        qc.h(i)
    
    for i in range(num_qubits-1):
        qc.cx(i, i+1)
    
    for i in range(num_qubits):
        qc.t(i)
    
    for i in range(num_qubits-1, 0, -1):
        qc.cx(i-1, i)
    
    for i in range(num_qubits):
        qc.h(i)
    
    return qc

def test_gnn_optimizer():
    """Test the GNN-based quantum circuit optimizer."""
    print("Creating test circuit...")
    circuit = create_test_circuit(num_qubits=3)
    
    print("Original circuit:")
    print(circuit)
    
    # Create a cost database
    cost_database = CostDatabase()
    
    # Create a GNN-based environment
    env = GNNQuantumEnvironment(
        initial_circuit=circuit,
        cost_database=cost_database,
        max_steps=50,
        depth_weight=0.3,
        cost_weight=0.3,
        error_weight=0.4,
        equivalence_bonus=1.0,
        node_feature_dim=16,
        render_mode=None,
        use_hierarchical_actions=True
    )
    
    # Create a hierarchical GNN-based agent
    agent = HierarchicalGNNDQNAgent(
        node_feature_dim=env.node_feature_dim,
        num_categories=env.num_categories,
        num_actions_per_category=env.num_actions_per_category,
        hidden_dim=64,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        buffer_size=1000,
        batch_size=32,
        target_update=10,
        gnn_type='gcn',
        num_gnn_layers=2,
        device='auto'
    )
    
    # Train the agent for a small number of steps (for testing)
    print("Training agent (brief training for testing)...")
    train_results = agent.train(env, num_steps=100)
    
    # Optimize the circuit
    print("Optimizing circuit...")
    state, _ = env.reset()
    
    for step in range(env.max_steps):
        action = agent.select_action(state, deterministic=True)
        next_state, reward, done, truncated, _ = env.step(action)
        state = next_state
        if done or truncated:
            break
    
    optimized_circuit = env.best_state.qiskit_circuit
    
    print("Optimized circuit:")
    print(optimized_circuit)
    
    # Calculate metrics
    metrics = CircuitMetrics()
    original_metrics = metrics.calculate_metrics(circuit)
    optimized_metrics = metrics.calculate_metrics(optimized_circuit, circuit)
    
    print("\nMetrics comparison:")
    print(f"Original depth: {original_metrics['depth']}")
    print(f"Optimized depth: {optimized_metrics['depth']}")
    print(f"Depth reduction: {optimized_metrics['depth_reduction']:.2f}%")
    
    print(f"Original gate count: {original_metrics['gate_count']}")
    print(f"Optimized gate count: {optimized_metrics['gate_count']}")
    print(f"Gate count reduction: {optimized_metrics['gate_count_reduction']:.2f}%")
    
    # Visualize the circuits
    visualizer = CircuitVisualizer()
    fig = visualizer.plot_circuit_comparison(circuit, optimized_circuit, 
                                           title="GNN Optimization Comparison")
    plt.show()
    
    return optimized_circuit, train_results

if __name__ == "__main__":
    test_gnn_optimizer()
