"""
GNN-based Quantum Circuit Optimizer Training
=========================================

This script trains a GNN-based quantum circuit optimizer using the hierarchical action space
and advanced reward mechanisms from the Quarl paper.
"""

import os
import argparse
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import QFT

from quantum_circuit_optimizer.quantum_circuit.circuit import QuantumCircuit as QCO
from quantum_circuit_optimizer.cost_database.cost_database import CostDatabase
from quantum_circuit_optimizer.environment.gnn_quantum_environment import GNNQuantumEnvironment
from quantum_circuit_optimizer.environment.advanced_reward import AdvancedReward
from quantum_circuit_optimizer.agent.gnn_dqn_agent import GNNDQNAgent, HierarchicalGNNDQNAgent
from quantum_circuit_optimizer.environment.hierarchical_action import HierarchicalCircuitAction, ActionCategory
from quantum_circuit_optimizer.evaluation.metrics import CircuitMetrics
from quantum_circuit_optimizer.evaluation.benchmarks import CircuitBenchmarks
from quantum_circuit_optimizer.evaluation.visualization import CircuitVisualizer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='GNN-based Quantum Circuit Optimizer')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'optimize', 'benchmark'],
                        help='Mode to run in (train, optimize, benchmark)')
    
    # Training parameters
    parser.add_argument('--agent', type=str, default='hierarchical',
                        choices=['gnn', 'hierarchical'],
                        help='RL agent to use (gnn, hierarchical)')
    parser.add_argument('--num_steps', type=int, default=20000,
                        help='Number of steps to train for')
    parser.add_argument('--save_path', type=str, default='models/gnn_agent.pt',
                        help='Path to save the trained agent')
    parser.add_argument('--load_path', type=str, default=None,
                        help='Path to load a trained agent from')
    
    # Circuit parameters
    parser.add_argument('--circuit_type', type=str, default='random',
                        choices=['random', 'qft', 'adder', 'grover', 'custom'],
                        help='Type of circuit to optimize')
    parser.add_argument('--num_qubits', type=int, default=5,
                        help='Number of qubits in the circuit')
    parser.add_argument('--depth', type=int, default=20,
                        help='Depth of the circuit (for random circuits)')
    parser.add_argument('--custom_circuit', type=str, default=None,
                        help='Path to a custom circuit file (QASM format)')
    
    # GNN parameters
    parser.add_argument('--gnn_type', type=str, default='gcn',
                        choices=['gcn', 'gat'],
                        help='Type of GNN to use')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension for GNN')
    parser.add_argument('--num_gnn_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--node_feature_dim', type=int, default=16,
                        help='Dimension of node features')
    
    # Optimization parameters
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Maximum number of optimization steps')
    parser.add_argument('--depth_weight', type=float, default=0.3,
                        help='Weight for depth in the reward function')
    parser.add_argument('--cost_weight', type=float, default=0.3,
                        help='Weight for cost in the reward function')
    parser.add_argument('--error_weight', type=float, default=0.4,
                        help='Weight for error in the reward function')
    parser.add_argument('--lookahead_factor', type=float, default=0.5,
                        help='Factor for weighing potential future improvements')
    
    # Benchmark parameters
    parser.add_argument('--num_circuits', type=int, default=10,
                        help='Number of circuits to benchmark')
    parser.add_argument('--min_qubits', type=int, default=3,
                        help='Minimum number of qubits for benchmark circuits')
    parser.add_argument('--max_qubits', type=int, default=7,
                        help='Maximum number of qubits for benchmark circuits')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    return parser.parse_args()

def create_circuit(args):
    """Create a circuit based on the arguments."""
    if args.circuit_type == 'random':
        # Create a random circuit
        circuit = random_circuit(args.num_qubits, args.depth, measure=False)
    elif args.circuit_type == 'qft':
        # Create a QFT circuit
        circuit = QFT(args.num_qubits)
    elif args.circuit_type == 'adder':
        # Create an adder circuit
        benchmarks = CircuitBenchmarks()
        circuit = benchmarks._create_quantum_adder(args.num_qubits)
    elif args.circuit_type == 'grover':
        # Create a Grover circuit
        benchmarks = CircuitBenchmarks()
        circuit = benchmarks._create_grover_circuit(args.num_qubits)
    elif args.circuit_type == 'custom':
        # Load a custom circuit from a QASM file
        if args.custom_circuit is None:
            raise ValueError("Must provide a custom circuit file with --custom_circuit")
        circuit = QuantumCircuit.from_qasm_file(args.custom_circuit)
    else:
        raise ValueError(f"Unknown circuit type: {args.circuit_type}")
    
    return circuit

def create_agent(args, env):
    """Create an RL agent based on the arguments."""
    # Get dimensions from environment
    node_feature_dim = args.node_feature_dim
    action_dim = env.action_space.n
    
    if args.agent == 'gnn':
        # Create a GNN-based DQN agent
        agent = GNNDQNAgent(
            node_feature_dim=node_feature_dim,
            action_dim=action_dim,
            hidden_dim=args.hidden_dim,
            learning_rate=1e-3,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.995,
            buffer_size=10000,
            batch_size=32,
            target_update=10,
            gnn_type=args.gnn_type,
            num_gnn_layers=args.num_gnn_layers,
            device='auto'
        )
    elif args.agent == 'hierarchical':
        # Create a hierarchical GNN-based DQN agent
        num_categories = env.num_categories
        num_actions_per_category = env.num_actions_per_category
        
        agent = HierarchicalGNNDQNAgent(
            node_feature_dim=node_feature_dim,
            num_categories=num_categories,
            num_actions_per_category=num_actions_per_category,
            hidden_dim=args.hidden_dim,
            learning_rate=1e-3,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.995,
            buffer_size=10000,
            batch_size=32,
            target_update=10,
            gnn_type=args.gnn_type,
            num_gnn_layers=args.num_gnn_layers,
            device='auto'
        )
    else:
        raise ValueError(f"Unknown agent type: {args.agent}")
    
    # Load the agent if a path is provided
    if args.load_path is not None:
        agent.load(args.load_path)
    
    return agent

def train_agent(args):
    """Train an RL agent for circuit optimization."""
    print("Training GNN-based RL agent for circuit optimization...")
    
    # Create a cost database
    cost_database = CostDatabase()
    
    # Create a circuit
    circuit = create_circuit(args)
    
    # Create an environment
    env = GNNQuantumEnvironment(
        initial_circuit=circuit,
        cost_database=cost_database,
        max_steps=args.max_steps,
        depth_weight=args.depth_weight,
        cost_weight=args.cost_weight,
        error_weight=args.error_weight,
        node_feature_dim=args.node_feature_dim,
        render_mode='human' if args.verbose else None,
        use_hierarchical_actions=(args.agent == 'hierarchical')
    )
    
    # Create an agent
    agent = create_agent(args, env)
    
    # Train the agent
    start_time = time.time()
    train_results = agent.train(env, args.num_steps)
    train_time = time.time() - start_time
    
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Save the agent
    if args.save_path is not None:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        agent.save(args.save_path)
        print(f"Agent saved to {args.save_path}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize training results
    visualizer = CircuitVisualizer()
    
    # Plot training rewards
    plt_rewards = visualizer.plot_training_rewards(
        train_results['episode_rewards'],
        title=f"Training Rewards ({args.agent} agent)",
        save_path=os.path.join(args.output_dir, f"{args.agent}_training_rewards.png")
    )
    
    # Plot training episode lengths
    plt_lengths = visualizer.plot_training_episode_lengths(
        train_results['episode_lengths'],
        title=f"Training Episode Lengths ({args.agent} agent)",
        save_path=os.path.join(args.output_dir, f"{args.agent}_training_lengths.png")
    )
    
    # Optimize a test circuit
    print("Optimizing a test circuit with the trained agent...")
    test_circuit = create_circuit(args)
    optimized_circuit = optimize_circuit_with_agent(test_circuit, agent, env, args)
    
    # Calculate metrics
    metrics = CircuitMetrics()
    original_metrics = metrics.calculate_metrics(test_circuit)
    optimized_metrics = metrics.calculate_metrics(optimized_circuit)
    
    print("Original circuit metrics:")
    for key, value in original_metrics.items():
        print(f"  {key}: {value}")
        
    print("Optimized circuit metrics:")
    for key, value in optimized_metrics.items():
        print(f"  {key}: {value}")
        
    # Visualize the optimization
    plt_comparison = visualizer.plot_circuit_comparison(
        test_circuit, optimized_circuit,
        title=f"Circuit Optimization ({args.agent} agent)",
        save_path=os.path.join(args.output_dir, f"{args.agent}_circuit_comparison.png")
    )
    
    return agent

def optimize_circuit_with_agent(circuit, agent, env, args):
    """
    Optimize a circuit using a trained agent.
    
    Args:
        circuit: The circuit to optimize.
        agent: The trained agent.
        env: The environment.
        args: Command line arguments.
        
    Returns:
        The optimized circuit.
    """
    # Reset the environment with the new circuit
    env.initial_circuit = circuit
    state, _ = env.reset()
    
    # Optimization loop
    for step in range(args.max_steps):
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
    
    return optimized_circuit

def optimize_circuit(args):
    """Optimize a quantum circuit using a trained agent."""
    print("Optimizing quantum circuit with GNN-based agent...")
    
    # Create a cost database
    cost_database = CostDatabase()
    
    # Create a circuit
    circuit = create_circuit(args)
    
    # Create an environment
    env = GNNQuantumEnvironment(
        initial_circuit=circuit,
        cost_database=cost_database,
        max_steps=args.max_steps,
        depth_weight=args.depth_weight,
        cost_weight=args.cost_weight,
        error_weight=args.error_weight,
        node_feature_dim=args.node_feature_dim,
        render_mode='human' if args.verbose else None,
        use_hierarchical_actions=(args.agent == 'hierarchical')
    )
    
    # Create or load an agent
    agent = create_agent(args, env)
    
    if args.load_path is None:
        print("No agent loaded. Training a new agent...")
        agent = train_agent(args)
    
    # Optimize the circuit
    start_time = time.time()
    optimized_circuit = optimize_circuit_with_agent(circuit, agent, env, args)
    optimize_time = time.time() - start_time
    
    print(f"Optimization completed in {optimize_time:.2f} seconds")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Calculate metrics
    metrics = CircuitMetrics()
    original_metrics = metrics.calculate_metrics(circuit)
    optimized_metrics = metrics.calculate_metrics(optimized_circuit)
    
    # Print metrics
    print("Original circuit metrics:")
    for key, value in original_metrics.items():
        print(f"  {key}: {value}")
        
    print("Optimized circuit metrics:")
    for key, value in optimized_metrics.items():
        print(f"  {key}: {value}")
        
    # Calculate improvement percentages
    depth_improvement = (original_metrics['depth'] - optimized_metrics['depth']) / original_metrics['depth'] * 100
    gate_improvement = (original_metrics['gate_count'] - optimized_metrics['gate_count']) / original_metrics['gate_count'] * 100
    
    print(f"Depth reduction: {depth_improvement:.2f}%")
    print(f"Gate count reduction: {gate_improvement:.2f}%")
    
    # Visualize the optimization
    visualizer = CircuitVisualizer()
    plt_comparison = visualizer.plot_circuit_comparison(
        circuit, optimized_circuit,
        title=f"Circuit Optimization ({args.agent} agent)",
        save_path=os.path.join(args.output_dir, f"{args.agent}_circuit_comparison.png")
    )
    
    # Save the optimized circuit
    from qiskit import qasm
    optimized_qasm = optimized_circuit.qasm()
    with open(os.path.join(args.output_dir, "optimized_circuit.qasm"), "w") as f:
        f.write(optimized_qasm)
    
    print(f"Optimized circuit saved to {os.path.join(args.output_dir, 'optimized_circuit.qasm')}")
    
    return optimized_circuit

def benchmark_optimizer(args):
    """Benchmark the quantum circuit optimizer."""
    print("Benchmarking GNN-based quantum circuit optimizer...")
    
    # Create a cost database
    cost_database = CostDatabase()
    
    # Create benchmarks
    benchmarks = CircuitBenchmarks()
    
    # Create an environment (with a placeholder circuit for now)
    placeholder_circuit = random_circuit(3, 5, measure=False)
    env = GNNQuantumEnvironment(
        initial_circuit=placeholder_circuit,
        cost_database=cost_database,
        max_steps=args.max_steps,
        depth_weight=args.depth_weight,
        cost_weight=args.cost_weight,
        error_weight=args.error_weight,
        node_feature_dim=args.node_feature_dim,
        render_mode=None,
        use_hierarchical_actions=(args.agent == 'hierarchical')
    )
    
    # Create or load an agent
    agent = create_agent(args, env)
    
    if args.load_path is None:
        print("No agent loaded. Training a new agent...")
        agent = train_agent(args)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Benchmark results
    results = {
        'circuit_type': [],
        'num_qubits': [],
        'original_depth': [],
        'optimized_depth': [],
        'depth_reduction': [],
        'original_gate_count': [],
        'optimized_gate_count': [],
        'gate_reduction': [],
        'optimization_time': []
    }
    
    # Run benchmarks
    circuit_types = ['random', 'qft', 'adder', 'grover']
    
    for circuit_type in circuit_types:
        print(f"Benchmarking {circuit_type} circuits...")
        
        for num_qubits in range(args.min_qubits, args.max_qubits + 1):
            print(f"  {num_qubits} qubits")
            
            # Create circuits
            if circuit_type == 'random':
                circuits = [random_circuit(num_qubits, args.depth, measure=False) for _ in range(args.num_circuits)]
            elif circuit_type == 'qft':
                circuits = [QFT(num_qubits) for _ in range(args.num_circuits)]
            elif circuit_type == 'adder':
                circuits = [benchmarks._create_quantum_adder(num_qubits) for _ in range(args.num_circuits)]
            elif circuit_type == 'grover':
                circuits = [benchmarks._create_grover_circuit(num_qubits) for _ in range(args.num_circuits)]
            else:
                continue
                
            # Optimize each circuit
            for i, circuit in enumerate(circuits):
                print(f"    Circuit {i+1}/{len(circuits)}")
                
                # Calculate original metrics
                metrics = CircuitMetrics()
                original_metrics = metrics.calculate_metrics(circuit)
                
                # Optimize the circuit
                start_time = time.time()
                env.initial_circuit = circuit
                optimized_circuit = optimize_circuit_with_agent(circuit, agent, env, args)
                optimize_time = time.time() - start_time
                
                # Calculate optimized metrics
                optimized_metrics = metrics.calculate_metrics(optimized_circuit)
                
                # Calculate improvement percentages
                depth_reduction = (original_metrics['depth'] - optimized_metrics['depth']) / max(1, original_metrics['depth']) * 100
                gate_reduction = (original_metrics['gate_count'] - optimized_metrics['gate_count']) / max(1, original_metrics['gate_count']) * 100
                
                # Store results
                results['circuit_type'].append(circuit_type)
                results['num_qubits'].append(num_qubits)
                results['original_depth'].append(original_metrics['depth'])
                results['optimized_depth'].append(optimized_metrics['depth'])
                results['depth_reduction'].append(depth_reduction)
                results['original_gate_count'].append(original_metrics['gate_count'])
                results['optimized_gate_count'].append(optimized_metrics['gate_count'])
                results['gate_reduction'].append(gate_reduction)
                results['optimization_time'].append(optimize_time)
    
    # Calculate average results
    avg_results = {}
    for circuit_type in circuit_types:
        avg_results[circuit_type] = {}
        for num_qubits in range(args.min_qubits, args.max_qubits + 1):
            avg_results[circuit_type][num_qubits] = {
                'depth_reduction': 0.0,
                'gate_reduction': 0.0,
                'optimization_time': 0.0,
                'count': 0
            }
    
    for i in range(len(results['circuit_type'])):
        circuit_type = results['circuit_type'][i]
        num_qubits = results['num_qubits'][i]
        avg_results[circuit_type][num_qubits]['depth_reduction'] += results['depth_reduction'][i]
        avg_results[circuit_type][num_qubits]['gate_reduction'] += results['gate_reduction'][i]
        avg_results[circuit_type][num_qubits]['optimization_time'] += results['optimization_time'][i]
        avg_results[circuit_type][num_qubits]['count'] += 1
    
    for circuit_type in circuit_types:
        for num_qubits in range(args.min_qubits, args.max_qubits + 1):
            count = avg_results[circuit_type][num_qubits]['count']
            if count > 0:
                avg_results[circuit_type][num_qubits]['depth_reduction'] /= count
                avg_results[circuit_type][num_qubits]['gate_reduction'] /= count
                avg_results[circuit_type][num_qubits]['optimization_time'] /= count
    
    # Print average results
    print("\nAverage benchmark results:")
    for circuit_type in circuit_types:
        print(f"\n{circuit_type.upper()} circuits:")
        print("  Qubits | Depth Reduction | Gate Reduction | Time (s)")
        print("  " + "-" * 50)
        for num_qubits in range(args.min_qubits, args.max_qubits + 1):
            if avg_results[circuit_type][num_qubits]['count'] > 0:
                print(f"  {num_qubits:6d} | {avg_results[circuit_type][num_qubits]['depth_reduction']:14.2f}% | {avg_results[circuit_type][num_qubits]['gate_reduction']:14.2f}% | {avg_results[circuit_type][num_qubits]['optimization_time']:8.2f}")
    
    # Visualize benchmark results
    visualizer = CircuitVisualizer()
    
    # Plot depth reduction
    plt_depth = visualizer.plot_benchmark_results(
        avg_results, 'depth_reduction',
        title=f"Depth Reduction ({args.agent} agent)",
        ylabel="Depth Reduction (%)",
        save_path=os.path.join(args.output_dir, f"{args.agent}_depth_reduction.png")
    )
    
    # Plot gate reduction
    plt_gate = visualizer.plot_benchmark_results(
        avg_results, 'gate_reduction',
        title=f"Gate Reduction ({args.agent} agent)",
        ylabel="Gate Reduction (%)",
        save_path=os.path.join(args.output_dir, f"{args.agent}_gate_reduction.png")
    )
    
    # Plot optimization time
    plt_time = visualizer.plot_benchmark_results(
        avg_results, 'optimization_time',
        title=f"Optimization Time ({args.agent} agent)",
        ylabel="Time (s)",
        save_path=os.path.join(args.output_dir, f"{args.agent}_optimization_time.png")
    )
    
    # Save benchmark results
    import json
    with open(os.path.join(args.output_dir, f"{args.agent}_benchmark_results.json"), "w") as f:
        json.dump({
            'results': results,
            'avg_results': avg_results
        }, f, indent=2)
    
    print(f"Benchmark results saved to {os.path.join(args.output_dir, f'{args.agent}_benchmark_results.json')}")

def main():
    """Main function."""
    args = parse_args()
    
    if args.mode == 'train':
        train_agent(args)
    elif args.mode == 'optimize':
        optimize_circuit(args)
    elif args.mode == 'benchmark':
        benchmark_optimizer(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == '__main__':
    main()
"""
