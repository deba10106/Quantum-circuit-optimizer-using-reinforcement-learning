"""
Focused test script for the GNN-based quantum circuit optimizer.

This script creates a quantum circuit with redundant gates (H-H pairs that cancel out)
and tests the GNN-based optimizer with improved debugging.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import time
from qiskit import QuantumCircuit

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
    
    return qc

def debug_action(action, env, success):
    """Debug an action by printing detailed information."""
    if isinstance(action, tuple):
        action_type = "Tuple"
        action_details = f"Category: {action[0]}, Action: {action[1]}"
    else:
        action_type = type(action).__name__
        try:
            action_details = f"Category: {action.category}, Action: {action.action}"
        except:
            action_details = str(action)
    
    result = "Success" if success else "Failed"
    print(f"Action {result}: Type={action_type}, {action_details}")

def focused_test():
    """Run a focused test on a circuit with redundant gates."""
    print("Creating redundant gate circuit...")
    circuit = create_redundant_gate_circuit(num_qubits=3)
    
    print("Original circuit:")
    print(circuit)
    
    # Create a cost database
    cost_database = CostDatabase()
    
    # Create an enhanced reward function with strong emphasis on gate reduction
    reward_function = ImprovedReward(
        depth_weight=0.3,
        cost_weight=0.2,
        error_weight=0.1,
        equivalence_bonus=2.0,   # Higher bonus for maintaining equivalence
        gate_reduction_weight=1.0, # Higher weight for gate reduction
        consecutive_improvement_bonus=0.5,
        exploration_bonus=0.2,
        lookahead_factor=0.8,
        history_length=5
    )
    
    # Create a GNN-based environment with the enhanced reward
    env = GNNQuantumEnvironment(
        initial_circuit=circuit,
        cost_database=cost_database,
        max_steps=30,  # Reduced max steps for this focused test
        depth_weight=reward_function.depth_weight,
        cost_weight=reward_function.cost_weight,
        error_weight=reward_function.error_weight,
        equivalence_bonus=reward_function.equivalence_bonus,
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
        learning_rate=1e-3,  # Slightly higher learning rate
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.99,  # Faster decay for this focused test
        buffer_size=2000,
        batch_size=32,
        target_update=50,
        gnn_type='gcn',
        num_gnn_layers=2,
        device='cpu'  # Force CPU to avoid CUDA issues
    )
    
    # Override the agent's step method to add debugging
    original_step = env.step
    
    def debug_step(action):
        try:
            next_state, reward, done, truncated, info = original_step(action)
            debug_action(action, env, True)
            return next_state, reward, done, truncated, info
        except Exception as e:
            debug_action(action, env, False)
            # Re-raise the exception to be handled by the caller
            raise e
    
    env.step = debug_step
    
    # Train the agent with a small number of steps
    print("\nTraining agent for 500 steps...")
    start_time = time.time()
    
    try:
        train_results = agent.train(env, num_steps=500)
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Plot training rewards
        plt.figure(figsize=(10, 6))
        plt.plot(train_results['rewards'])
        plt.title('Training Rewards for Redundant Gate Circuit')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig("focused_test_rewards.png")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        print("Proceeding with optimization using the partially trained agent")
        training_time = time.time() - start_time
    
    # Optimize the circuit with detailed logging
    print("\nOptimizing circuit...")
    state, _ = env.reset()
    
    # Restore original step method for optimization
    env.step = original_step
    
    # Track actions and states for analysis
    action_history = []
    state_history = []
    reward_history = []
    
    try:
        for step in range(env.max_steps):
            print(f"\nStep {step+1}/{env.max_steps}")
            action = agent.select_action(state, deterministic=True)
            action_history.append(action)
            
            try:
                next_state, reward, done, truncated, _ = env.step(action)
                print(f"Action successful, reward: {reward:.4f}")
                state = next_state
                state_history.append(state)
                reward_history.append(reward)
                
                # Print current circuit state
                print(f"Current circuit depth: {state.depth}")
                print(f"Current circuit gate count: {len(state.circuit_dag.nodes())}")
                
                if done or truncated:
                    print("Optimization complete!")
                    break
            except Exception as e:
                print(f"Action failed: {str(e)}")
        
        optimized_circuit = env.best_state.qiskit_circuit
        
        print("\nOptimized circuit:")
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
                                                  title="GNN Optimization for Redundant Gate Circuit")
            plt.savefig("focused_test_circuit_comparison.png")
            plt.show()
        except Exception as e:
            print(f"Error during visualization: {str(e)}")
        
        # Analyze action history
        if action_history:
            print("\nAction history analysis:")
            action_types = {}
            for action in action_history:
                if hasattr(action, 'category'):
                    category = str(action.category)
                    if category not in action_types:
                        action_types[category] = 0
                    action_types[category] += 1
            
            print("Action categories used:")
            for category, count in action_types.items():
                print(f"  {category}: {count} times")
        
        return optimized_circuit, train_results
    
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        return circuit, None

if __name__ == "__main__":
    focused_test()
