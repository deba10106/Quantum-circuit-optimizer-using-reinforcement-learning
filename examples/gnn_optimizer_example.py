"""
GNN-based Quantum Circuit Optimizer Example
=======================================

This example demonstrates how to use the GNN-based quantum circuit optimizer
with hierarchical action space and advanced reward mechanisms.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import QFT

# Import our modules
from quantum_circuit.dag import CircuitDAG
from environment.gnn_quantum_environment import GNNQuantumEnvironment
from environment.advanced_reward import AdvancedReward, EquivalenceChecker
from agent.gnn_dqn_agent import GNNDQNAgent
from agent.hierarchical_gnn_dqn_agent import HierarchicalGNNDQNAgent
from evaluation.metrics import CircuitMetrics
from evaluation.visualization import CircuitVisualizer
from evaluation.gnn_analysis import GNNOptimizationAnalyzer
from cost_database.cost_database import CostDatabase

def create_example_circuit(num_qubits=5, depth=20, circuit_type='random'):
    """Create an example quantum circuit."""
    if circuit_type == 'random':
        return random_circuit(num_qubits, depth, measure=False)
    elif circuit_type == 'qft':
        return QFT(num_qubits)
    else:
        raise ValueError(f"Unknown circuit type: {circuit_type}")

def optimize_with_gnn_dqn(circuit, num_steps=1000, use_hierarchical=True, verbose=True):
    """
    Optimize a quantum circuit using GNN-based DQN.
    
    Args:
        circuit: The circuit to optimize.
        num_steps: Number of training steps.
        use_hierarchical: Whether to use hierarchical action space.
        verbose: Whether to print progress.
        
    Returns:
        The optimized circuit and training results.
    """
    # Create a cost database
    cost_database = CostDatabase()
    
    # Create an environment
    env = GNNQuantumEnvironment(
        initial_circuit=circuit,
        cost_database=cost_database,
        max_steps=100,
        depth_weight=0.3,
        cost_weight=0.3,
        error_weight=0.4,
        equivalence_bonus=1.0,
        node_feature_dim=16,
        render_mode='human' if verbose else None,
        use_hierarchical_actions=use_hierarchical
    )
    
    # Get dimensions from environment
    node_feature_dim = env.node_feature_dim
    
    # Create an agent
    if use_hierarchical:
        # Create a hierarchical GNN-based DQN agent
        num_categories = env.num_categories
        num_actions_per_category = env.num_actions_per_category
        
        agent = HierarchicalGNNDQNAgent(
            node_feature_dim=node_feature_dim,
            num_categories=num_categories,
            num_actions_per_category=num_actions_per_category,
            hidden_dim=64,
            learning_rate=1e-3,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.995,
            buffer_size=10000,
            batch_size=32,
            target_update=10,
            gnn_type='gcn',
            num_gnn_layers=3,
            device='auto'
        )
    else:
        # Create a standard GNN-based DQN agent
        action_dim = env.action_space.n
        
        agent = GNNDQNAgent(
            node_feature_dim=node_feature_dim,
            action_dim=action_dim,
            hidden_dim=64,
            learning_rate=1e-3,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.995,
            buffer_size=10000,
            batch_size=32,
            target_update=10,
            gnn_type='gcn',
            num_gnn_layers=3,
            device='auto'
        )
    
    # Train the agent
    if verbose:
        print(f"Training {'hierarchical ' if use_hierarchical else ''}GNN-based DQN agent...")
    
    train_results = agent.train(env, num_steps)
    
    # Optimize the circuit
    if verbose:
        print("Optimizing circuit with trained agent...")
    
    # Reset the environment
    state, _ = env.reset()
    
    # Optimization loop
    for step in range(env.max_steps):
        # Select an action
        action = agent.select_action(state, deterministic=True)
        
        # Apply the action
        next_state, reward, done, truncated, _ = env.step(action)
        
        # Move to the next state
        state = next_state
        
        # Check if done
        if done or truncated:
            break
    
    # Get the best circuit found
    optimized_circuit = env.best_state.qiskit_circuit
    
    return optimized_circuit, train_results, agent, env

def main():
    """Main function."""
    # Create output directory
    os.makedirs('results', exist_ok=True)
    
    # Create an example circuit
    print("Creating example circuit...")
    circuit = create_example_circuit(num_qubits=5, depth=20, circuit_type='random')
    
    # Print circuit information
    print(f"Circuit qubits: {circuit.num_qubits}")
    print(f"Circuit depth: {circuit.depth()}")
    print(f"Circuit gate count: {sum(1 for op in circuit.data if op[0].name not in ['barrier', 'snapshot'])}")
    
    # Optimize with hierarchical GNN-based DQN
    print("\nOptimizing with hierarchical GNN-based DQN...")
    optimized_circuit_h, train_results_h, agent_h, env_h = optimize_with_gnn_dqn(
        circuit, num_steps=1000, use_hierarchical=True, verbose=True
    )
    
    # Optimize with standard GNN-based DQN
    print("\nOptimizing with standard GNN-based DQN...")
    optimized_circuit_s, train_results_s, agent_s, env_s = optimize_with_gnn_dqn(
        circuit, num_steps=1000, use_hierarchical=False, verbose=True
    )
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = CircuitMetrics()
    
    original_metrics = metrics.calculate_metrics(circuit)
    optimized_metrics_h = metrics.calculate_metrics(optimized_circuit_h, circuit)
    optimized_metrics_s = metrics.calculate_metrics(optimized_circuit_s, circuit)
    
    # Print metrics
    print("\nOriginal circuit metrics:")
    for key, value in original_metrics.items():
        if key not in ['gate_counts', 'original_gate_counts']:
            print(f"  {key}: {value}")
    
    print("\nHierarchical GNN-DQN optimized circuit metrics:")
    for key, value in optimized_metrics_h.items():
        if key not in ['gate_counts', 'original_gate_counts'] and key.startswith('depth') or key.startswith('gate'):
            print(f"  {key}: {value}")
    
    print("\nStandard GNN-DQN optimized circuit metrics:")
    for key, value in optimized_metrics_s.items():
        if key not in ['gate_counts', 'original_gate_counts'] and key.startswith('depth') or key.startswith('gate'):
            print(f"  {key}: {value}")
    
    # Visualize results
    print("\nVisualizing results...")
    visualizer = CircuitVisualizer()
    
    # Plot circuits
    visualizer.plot_circuit(circuit, title="Original Circuit", 
                           filename="results/original_circuit.png")
    
    visualizer.plot_circuit(optimized_circuit_h, title="Hierarchical GNN-DQN Optimized Circuit", 
                           filename="results/hierarchical_optimized_circuit.png")
    
    visualizer.plot_circuit(optimized_circuit_s, title="Standard GNN-DQN Optimized Circuit", 
                           filename="results/standard_optimized_circuit.png")
    
    # Plot circuit comparison
    visualizer.plot_circuit_comparison(circuit, optimized_circuit_h, 
                                     title="Hierarchical GNN-DQN Optimization", 
                                     filename="results/hierarchical_circuit_comparison.png")
    
    visualizer.plot_circuit_comparison(circuit, optimized_circuit_s, 
                                     title="Standard GNN-DQN Optimization", 
                                     filename="results/standard_circuit_comparison.png")
    
    # Plot optimization metrics
    visualizer.plot_optimization_metrics(optimized_metrics_h, 
                                       title="Hierarchical GNN-DQN Optimization Metrics", 
                                       filename="results/hierarchical_optimization_metrics.png")
    
    visualizer.plot_optimization_metrics(optimized_metrics_s, 
                                       title="Standard GNN-DQN Optimization Metrics", 
                                       filename="results/standard_optimization_metrics.png")
    
    # Plot training rewards
    visualizer.plot_training_rewards(train_results_h['episode_rewards'], 
                                   title="Hierarchical GNN-DQN Training Rewards", 
                                   save_path="results/hierarchical_training_rewards.png")
    
    visualizer.plot_training_rewards(train_results_s['episode_rewards'], 
                                   title="Standard GNN-DQN Training Rewards", 
                                   save_path="results/standard_training_rewards.png")
    
    # Advanced analysis
    print("\nPerforming advanced analysis...")
    analyzer = GNNOptimizationAnalyzer()
    
    # Visualize circuit graph
    analyzer.visualize_circuit_graph(circuit, title="Circuit Graph", 
                                   save_path="results/circuit_graph.png")
    
    # Analyze action distribution for hierarchical agent
    analyzer.analyze_action_distribution(agent_h, env_h, num_episodes=5, 
                                      title="Hierarchical Action Distribution", 
                                      save_path="results/hierarchical_action_distribution.png")
    
    # Analyze optimization trajectory
    analyzer.analyze_optimization_trajectory(agent_h, env_h, 
                                          title="Hierarchical Optimization Trajectory", 
                                          save_path="results/hierarchical_optimization_trajectory.png")
    
    # Compare optimization methods
    circuits = [create_example_circuit(num_qubits=4, depth=15) for _ in range(5)]
    
    def optimize_with_hierarchical(circuit):
        opt_circuit, _, _, _ = optimize_with_gnn_dqn(circuit, num_steps=500, use_hierarchical=True, verbose=False)
        return opt_circuit
    
    def optimize_with_standard(circuit):
        opt_circuit, _, _, _ = optimize_with_gnn_dqn(circuit, num_steps=500, use_hierarchical=False, verbose=False)
        return opt_circuit
    
    analyzer.compare_optimization_methods(
        circuits, 
        [optimize_with_hierarchical, optimize_with_standard], 
        labels=['Hierarchical', 'Standard'],
        metric='depth',
        title="Comparison of Optimization Methods",
        save_path="results/optimization_methods_comparison.png"
    )
    
    print("\nExample completed. Results saved to 'results' directory.")

if __name__ == "__main__":
    main()
