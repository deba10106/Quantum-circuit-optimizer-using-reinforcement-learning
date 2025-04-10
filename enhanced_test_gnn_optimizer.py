"""
Enhanced test script for the GNN-based quantum circuit optimizer.

This script creates multiple quantum circuits with known optimization opportunities
and tests the GNN-based optimizer with extended training.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import time
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT

from quantum_circuit.dag import CircuitDAG
from environment.gnn_quantum_environment import GNNQuantumEnvironment
from agent.hierarchical_gnn_dqn_agent import HierarchicalGNNDQNAgent
from cost_database.cost_database import CostDatabase
from evaluation.metrics import CircuitMetrics
from evaluation.visualization import CircuitVisualizer
from environment.improved_reward import ImprovedReward

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

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
    
    # Add some more gates
    for i in range(num_qubits-1):
        qc.cx(i, i+1)
    
    # Add redundant CNOT gates (CNOT-CNOT cancels to identity)
    for i in range(num_qubits-1):
        qc.cx(i, i+1)
        qc.cx(i, i+1)
    
    # Add some more gates
    for i in range(num_qubits):
        qc.t(i)
    
    return qc

def create_commutable_gate_circuit(num_qubits=3):
    """Create a circuit with gates that can be commuted to reduce depth."""
    qc = QuantumCircuit(num_qubits)
    
    # Add initial layer
    for i in range(num_qubits):
        qc.h(i)
    
    # Add CX gates in a pattern that can be optimized by reordering
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(0, 2)  # This can be reordered to reduce depth
    
    # Add single-qubit gates that can be commuted
    qc.t(0)
    qc.t(1)
    qc.t(2)
    
    # Add more CX gates
    qc.cx(0, 1)
    qc.cx(1, 2)
    
    # Final layer
    for i in range(num_qubits):
        qc.h(i)
    
    return qc

def create_decomposable_gate_circuit(num_qubits=3):
    """Create a circuit with gates that can be decomposed into simpler ones."""
    qc = QuantumCircuit(num_qubits)
    
    # Add some gates
    for i in range(num_qubits):
        qc.h(i)
    
    # Add SWAP gates (which can be decomposed into 3 CNOT gates)
    for i in range(num_qubits-1):
        qc.swap(i, i+1)
    
    # Add some more gates
    for i in range(num_qubits):
        qc.t(i)
    
    # Add more SWAP gates
    for i in range(num_qubits-1, 0, -1):
        qc.swap(i-1, i)
    
    # Final layer
    for i in range(num_qubits):
        qc.h(i)
    
    return qc

def create_qft_circuit(num_qubits=3):
    """Create a QFT circuit which has known optimization opportunities."""
    return QFT(num_qubits)

def create_circuit_suite():
    """Create a suite of test circuits with known optimization opportunities."""
    circuits = {
        "redundant_gates": create_redundant_gate_circuit(3),
        "commutable_gates": create_commutable_gate_circuit(3),
        "decomposable_gates": create_decomposable_gate_circuit(3),
        "qft": create_qft_circuit(3)
    }
    return circuits

def enhanced_reward_function():
    """Create an enhanced reward function with better guidance for optimization."""
    return ImprovedReward(
        depth_weight=0.4,        # Increased weight for depth reduction
        cost_weight=0.3,
        error_weight=0.2,        # Reduced weight for error to allow more exploration
        equivalence_bonus=1.5,   # Increased bonus for maintaining equivalence
        gate_reduction_weight=0.5, # Weight for gate reduction
        consecutive_improvement_bonus=0.2, # Bonus for consecutive improvements
        exploration_bonus=0.1,   # Bonus for exploring new circuit configurations
        lookahead_factor=0.7,    # Increased lookahead to encourage long-term planning
        history_length=10        # Longer history for better trend analysis
    )

def train_and_test_optimizer(circuit, circuit_name, num_training_steps=1000, save_dir="results"):
    """Train and test the GNN-based optimizer on a specific circuit."""
    print(f"\n{'='*50}")
    print(f"Testing on {circuit_name} circuit")
    print(f"{'='*50}")
    
    print("Original circuit:")
    print(circuit)
    
    # Create a cost database
    cost_database = CostDatabase()
    
    # Create an enhanced reward function
    reward_function = enhanced_reward_function()
    
    # Create a GNN-based environment with the enhanced reward
    env = GNNQuantumEnvironment(
        initial_circuit=circuit,
        cost_database=cost_database,
        max_steps=50,
        depth_weight=reward_function.depth_weight,
        cost_weight=reward_function.cost_weight,
        error_weight=reward_function.error_weight,
        equivalence_bonus=reward_function.equivalence_bonus,
        node_feature_dim=16,  # Using default feature dimension to avoid memory issues
        render_mode=None,
        use_hierarchical_actions=True
    )
    
    # Create a hierarchical GNN-based agent with improved parameters
    agent = HierarchicalGNNDQNAgent(
        node_feature_dim=env.node_feature_dim,
        num_categories=env.num_categories,
        num_actions_per_category=env.num_actions_per_category,
        hidden_dim=64,  # Using smaller hidden dimension to avoid memory issues
        learning_rate=5e-4,  # Adjusted learning rate
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,  # Lower end epsilon for better exploitation
        epsilon_decay=0.995,  # Slower decay
        buffer_size=5000,  # Reduced buffer size to avoid memory issues
        batch_size=32,  # Smaller batch size
        target_update=100,  # More frequent target updates
        gnn_type='gcn',
        num_gnn_layers=2,  # Reduced number of GNN layers
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train the agent with more steps
    print(f"Training agent for {num_training_steps} steps...")
    start_time = time.time()
    
    try:
        train_results = agent.train(env, num_steps=num_training_steps)
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save training results
        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/{circuit_name}_train_results.npy", train_results)
        
        # Plot training rewards
        plt.figure(figsize=(10, 6))
        plt.plot(train_results['rewards'])
        plt.title(f'Training Rewards for {circuit_name} Circuit')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig(f"{save_dir}/{circuit_name}_training_rewards.png")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        print("Proceeding with optimization using the partially trained agent")
        training_time = time.time() - start_time
    
    # Optimize the circuit
    print("Optimizing circuit...")
    state, _ = env.reset()
    
    try:
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
        try:
            visualizer = CircuitVisualizer()
            fig = visualizer.plot_circuit_comparison(circuit, optimized_circuit, 
                                                   title=f"GNN Optimization for {circuit_name}")
            plt.savefig(f"{save_dir}/{circuit_name}_circuit_comparison.png")
        except Exception as e:
            print(f"Error during visualization: {str(e)}")
        
        # Save detailed metrics
        metrics_dict = {
            "original_depth": original_metrics['depth'],
            "optimized_depth": optimized_metrics['depth'],
            "depth_reduction": optimized_metrics['depth_reduction'],
            "original_gate_count": original_metrics['gate_count'],
            "optimized_gate_count": optimized_metrics['gate_count'],
            "gate_count_reduction": optimized_metrics['gate_count_reduction'],
            "training_time": training_time,
            "training_steps": num_training_steps
        }
        np.save(f"{save_dir}/{circuit_name}_metrics.npy", metrics_dict)
        
        return optimized_circuit, train_results, metrics_dict
    
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        # Return default values if optimization fails
        return circuit, None, {
            "original_depth": 0,
            "optimized_depth": 0,
            "depth_reduction": 0,
            "original_gate_count": 0,
            "optimized_gate_count": 0,
            "gate_count_reduction": 0,
            "training_time": training_time,
            "training_steps": num_training_steps
        }

def run_enhanced_tests(num_training_steps=1000, save_dir="results"):
    """Run enhanced tests on multiple circuits."""
    # Create directory for results
    os.makedirs(save_dir, exist_ok=True)
    
    # Create test circuits
    circuits = create_circuit_suite()
    
    # Dictionary to store all results
    all_results = {}
    
    # Test each circuit
    for circuit_name, circuit in circuits.items():
        try:
            _, _, metrics = train_and_test_optimizer(
                circuit, 
                circuit_name, 
                num_training_steps=num_training_steps,
                save_dir=save_dir
            )
            all_results[circuit_name] = metrics
        except Exception as e:
            print(f"Error testing {circuit_name} circuit: {str(e)}")
            all_results[circuit_name] = {
                "depth_reduction": 0,
                "gate_count_reduction": 0
            }
    
    # Generate summary report
    print("\n" + "="*50)
    print("SUMMARY OF RESULTS")
    print("="*50)
    
    print("\nDepth Reduction:")
    for circuit_name, metrics in all_results.items():
        print(f"{circuit_name}: {metrics.get('depth_reduction', 0):.2f}%")
    
    print("\nGate Count Reduction:")
    for circuit_name, metrics in all_results.items():
        print(f"{circuit_name}: {metrics.get('gate_count_reduction', 0):.2f}%")
    
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
    
    # Save all results
    np.save(f"{save_dir}/all_results.npy", all_results)
    
    return all_results

if __name__ == "__main__":
    # Use fewer training steps to avoid CUDA memory issues
    run_enhanced_tests(num_training_steps=1000, save_dir="enhanced_results")
