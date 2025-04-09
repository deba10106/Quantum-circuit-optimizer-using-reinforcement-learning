"""
Quantum Circuit Optimizer
======================

Main script for training and using the reinforcement learning-enabled quantum circuit optimizer.
"""

import os
import argparse
import time
import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import QFT
from qiskit import Aer, execute

from quantum_circuit_optimizer.quantum_circuit.circuit import QuantumCircuit as QCO
from quantum_circuit_optimizer.cost_database.cost_database import CostDatabase
from quantum_circuit_optimizer.environment.quantum_environment import QuantumEnvironment
from quantum_circuit_optimizer.agent.dqn_agent import DQNAgent
from quantum_circuit_optimizer.agent.ppo_agent import PPOAgent
from quantum_circuit_optimizer.evaluation.metrics import CircuitMetrics
from quantum_circuit_optimizer.evaluation.benchmarks import CircuitBenchmarks
from quantum_circuit_optimizer.evaluation.visualization import CircuitVisualizer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Quantum Circuit Optimizer')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'optimize', 'benchmark'],
                        help='Mode to run in (train, optimize, benchmark)')
    
    # Training parameters
    parser.add_argument('--agent', type=str, default='dqn',
                        choices=['dqn', 'ppo'],
                        help='RL agent to use (dqn, ppo)')
    parser.add_argument('--num_steps', type=int, default=10000,
                        help='Number of steps to train for')
    parser.add_argument('--save_path', type=str, default='models/agent.pt',
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
    
    # Optimization parameters
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Maximum number of optimization steps')
    parser.add_argument('--depth_weight', type=float, default=0.3,
                        help='Weight for depth in the reward function')
    parser.add_argument('--cost_weight', type=float, default=0.3,
                        help='Weight for cost in the reward function')
    parser.add_argument('--error_weight', type=float, default=0.4,
                        help='Weight for error in the reward function')
    
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

def create_agent(args, state_dim, action_dim):
    """Create an RL agent based on the arguments."""
    if args.agent == 'dqn':
        # Create a DQN agent
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
            device='auto'
        )
    elif args.agent == 'ppo':
        # Create a PPO agent
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_param=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            ppo_epochs=10,
            batch_size=64,
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
    print("Training RL agent for circuit optimization...")
    
    # Create a cost database
    cost_database = CostDatabase()
    
    # Create a circuit
    circuit = create_circuit(args)
    
    # Create an environment
    env = QuantumEnvironment(
        initial_circuit=circuit,
        cost_database=cost_database,
        max_steps=args.max_steps,
        depth_weight=args.depth_weight,
        cost_weight=args.cost_weight,
        error_weight=args.error_weight,
        feature_dim=32,
        render_mode='human' if args.verbose else None
    )
    
    # Create an agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = create_agent(args, state_dim, action_dim)
    
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
    plt_rewards = visualizer.plot_optimization_progress(
        train_results['rewards'],
        [{'depth': 0, 'gate_count': 0}] * len(train_results['rewards']),  # Dummy metrics
        title='Training Rewards',
        filename=os.path.join(args.output_dir, 'training_rewards.png')
    )
    
    # Evaluate the agent
    print("Evaluating agent...")
    eval_results = agent.evaluate(env, num_episodes=5, render=args.verbose)
    
    print(f"Evaluation results:")
    print(f"  Mean reward: {eval_results['mean_reward']:.4f} ± {eval_results['std_reward']:.4f}")
    print(f"  Mean length: {eval_results['mean_length']:.2f} ± {eval_results['std_length']:.2f}")
    
    return agent

def optimize_circuit(args, agent=None):
    """Optimize a quantum circuit using a trained agent."""
    print("Optimizing quantum circuit...")
    
    # Create a cost database
    cost_database = CostDatabase()
    
    # Create a circuit
    circuit = create_circuit(args)
    
    # Create an environment
    env = QuantumEnvironment(
        initial_circuit=circuit,
        cost_database=cost_database,
        max_steps=args.max_steps,
        depth_weight=args.depth_weight,
        cost_weight=args.cost_weight,
        error_weight=args.error_weight,
        feature_dim=32,
        render_mode='human' if args.verbose else None
    )
    
    # Create or load an agent
    if agent is None:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        agent = create_agent(args, state_dim, action_dim)
    
    # Optimize the circuit
    start_time = time.time()
    optimized_circuit, optimization_info = agent.optimize_circuit(
        env, circuit, max_steps=args.max_steps, render=args.verbose
    )
    optimization_time = time.time() - start_time
    
    print(f"Optimization completed in {optimization_time:.2f} seconds")
    
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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check circuit equivalence
    metrics = CircuitMetrics(cost_database)
    equivalent = metrics.check_equivalence(circuit, optimized_circuit.circuit)
    print(f"Circuits are equivalent: {equivalent}")
    
    # Visualize results
    visualizer = CircuitVisualizer()
    
    # Plot circuit comparison
    plt_comparison = visualizer.plot_circuit_comparison(
        circuit, optimized_circuit.circuit,
        title='Circuit Comparison',
        filename=os.path.join(args.output_dir, 'circuit_comparison.png')
    )
    
    # Plot optimization metrics
    metrics_dict = metrics.calculate_metrics(optimized_circuit.circuit, circuit)
    plt_metrics = visualizer.plot_optimization_metrics(
        metrics_dict,
        title='Optimization Metrics',
        filename=os.path.join(args.output_dir, 'optimization_metrics.png')
    )
    
    # Plot gate distributions
    plt_gates_original = visualizer.plot_gate_distribution(
        circuit,
        title='Original Circuit Gate Distribution',
        filename=os.path.join(args.output_dir, 'original_gate_distribution.png')
    )
    
    plt_gates_optimized = visualizer.plot_gate_distribution(
        optimized_circuit.circuit,
        title='Optimized Circuit Gate Distribution',
        filename=os.path.join(args.output_dir, 'optimized_gate_distribution.png')
    )
    
    # Save circuits to QASM files
    original_qasm = circuit.qasm()
    optimized_qasm = optimized_circuit.circuit.qasm()
    
    with open(os.path.join(args.output_dir, 'original_circuit.qasm'), 'w') as f:
        f.write(original_qasm)
    
    with open(os.path.join(args.output_dir, 'optimized_circuit.qasm'), 'w') as f:
        f.write(optimized_qasm)
    
    print(f"Results saved to {args.output_dir}")
    
    return optimized_circuit, optimization_info

def benchmark_optimizer(args):
    """Benchmark the quantum circuit optimizer."""
    print("Benchmarking quantum circuit optimizer...")
    
    # Create a cost database
    cost_database = CostDatabase()
    
    # Create benchmark circuits
    benchmarks = CircuitBenchmarks(cost_database)
    
    if args.circuit_type == 'random':
        # Generate random benchmark circuits
        circuits = benchmarks.generate_random_circuits(
            num_circuits=args.num_circuits,
            min_qubits=args.min_qubits,
            max_qubits=args.max_qubits,
            min_depth=5,
            max_depth=30,
            seed=42
        )
    elif args.circuit_type == 'qft':
        # Generate QFT benchmark circuits
        circuits = benchmarks.generate_qft_circuits(
            min_qubits=args.min_qubits,
            max_qubits=args.max_qubits
        )
    elif args.circuit_type == 'adder':
        # Generate adder benchmark circuits
        circuits = benchmarks.generate_adder_circuits(
            min_bits=args.min_qubits,
            max_bits=args.max_qubits
        )
    elif args.circuit_type == 'grover':
        # Generate Grover benchmark circuits
        circuits = benchmarks.generate_grover_circuits(
            min_qubits=args.min_qubits,
            max_qubits=args.max_qubits
        )
    else:
        raise ValueError(f"Unknown circuit type: {args.circuit_type}")
    
    # Create an environment
    env = QuantumEnvironment(
        initial_circuit=circuits[0],
        cost_database=cost_database,
        max_steps=args.max_steps,
        depth_weight=args.depth_weight,
        cost_weight=args.cost_weight,
        error_weight=args.error_weight,
        feature_dim=32,
        render_mode=None
    )
    
    # Create agents
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agents = {}
    
    # Create DQN agent
    dqn_agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=0.1,  # Low epsilon for evaluation
        epsilon_end=0.1,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update=10,
        device='auto'
    )
    
    # Create PPO agent
    ppo_agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_param=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=10,
        batch_size=64,
        device='auto'
    )
    
    # Load agents if paths are provided
    if args.load_path is not None:
        if args.agent == 'dqn':
            dqn_agent.load(args.load_path)
            agents['DQN'] = dqn_agent
        elif args.agent == 'ppo':
            ppo_agent.load(args.load_path)
            agents['PPO'] = ppo_agent
    else:
        # Use both agents for comparison
        agents['DQN'] = dqn_agent
        agents['PPO'] = ppo_agent
    
    # Create optimizer wrappers
    class OptimizerWrapper:
        def __init__(self, agent, env, max_steps):
            self.agent = agent
            self.env = env
            self.max_steps = max_steps
        
        def optimize(self, circuit):
            self.env.initial_circuit = circuit
            self.env.max_steps = self.max_steps
            optimized_circuit, _ = self.agent.optimize_circuit(
                self.env, circuit, max_steps=self.max_steps, render=False
            )
            return optimized_circuit
    
    optimizers = {}
    for name, agent in agents.items():
        optimizers[name] = OptimizerWrapper(agent, env, args.max_steps)
    
    # Benchmark optimizers
    comparison = benchmarks.compare_optimizers(optimizers, circuits, verbose=args.verbose)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize benchmark results
    visualizer = CircuitVisualizer()
    
    # Plot benchmark results for each optimizer
    for name, results in comparison.items():
        plt_results = visualizer.plot_benchmark_results(
            results,
            title=f'{name} Benchmark Results',
            filename=os.path.join(args.output_dir, f'{name.lower()}_benchmark_results.png')
        )
    
    # Plot optimizer comparison
    metrics_to_compare = [
        'mean_depth_reduction_percent',
        'mean_gate_count_reduction_percent',
        'mean_optimization_time',
        'equivalence_rate'
    ]
    
    for metric in metrics_to_compare:
        plt_comparison = visualizer.plot_optimizer_comparison(
            comparison,
            metric=metric,
            title=f'Optimizer Comparison - {metric.replace("_", " ").title()}',
            filename=os.path.join(args.output_dir, f'optimizer_comparison_{metric}.png')
        )
    
    print(f"Benchmark results saved to {args.output_dir}")
    
    return comparison

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Run in the specified mode
    if args.mode == 'train':
        # Train an agent
        agent = train_agent(args)
        
        # Optimize a circuit with the trained agent
        if args.verbose:
            optimize_circuit(args, agent)
    
    elif args.mode == 'optimize':
        # Optimize a circuit
        optimize_circuit(args)
    
    elif args.mode == 'benchmark':
        # Benchmark optimizers
        benchmark_optimizer(args)
    
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == '__main__':
    main()
