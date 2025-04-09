"""
Example: Benchmark Circuits
=========================

This example demonstrates how to benchmark the quantum circuit optimizer
on different types of quantum circuits.
"""

import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT, RealAmplitudes, EfficientSU2
from qiskit.circuit.random import random_circuit

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the required modules
from quantum_circuit_optimizer.quantum_circuit.circuit import QuantumCircuit as QCO
from quantum_circuit_optimizer.cost_database.cost_database import CostDatabase
from quantum_circuit_optimizer.environment.quantum_environment import QuantumEnvironment
from quantum_circuit_optimizer.agent.dqn_agent import DQNAgent
from quantum_circuit_optimizer.evaluation.metrics import CircuitMetrics
from quantum_circuit_optimizer.evaluation.visualization import CircuitVisualizer
from quantum_circuit_optimizer.evaluation.benchmarks import CircuitBenchmarks

def optimize_circuit(circuit, name, agent=None, steps=1000, max_steps=50):
    """Optimize a circuit and print the results."""
    print(f"\n=== Optimizing {name} ===")
    
    # Create a cost database
    cost_database = CostDatabase()
    
    # Print the original circuit
    print(f"Original {name} circuit:")
    print(circuit)
    print(f"Original depth: {circuit.depth()}")
    print(f"Original gate count: {len(circuit.data)}")
    
    # Create an environment
    print("Creating the RL environment...")
    env = QuantumEnvironment(
        initial_circuit=circuit,
        cost_database=cost_database,
        max_steps=max_steps,
        depth_weight=0.3,
        cost_weight=0.3,
        error_weight=0.4,
        feature_dim=32,
        render_mode=None
    )
    
    # Create a DQN agent if not provided
    if agent is None:
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
            device='cpu'  # Force CPU usage
        )
        
        # Train the agent
        print(f"Training the agent for {steps} steps...")
        start_time = time.time()
        train_results = agent.train(env, steps)
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds")
    
    # Optimize the circuit
    print("Optimizing the circuit...")
    start_time = time.time()
    optimized_circuit, optimization_info = agent.optimize_circuit(
        env, circuit, max_steps=max_steps, render=False
    )
    optimize_time = time.time() - start_time
    print(f"Optimization completed in {optimize_time:.2f} seconds")
    
    # Print optimization results
    print("Optimization results:")
    print(f"  Original depth: {optimization_info['initial_depth']}")
    print(f"  Optimized depth: {optimization_info['optimized_depth']}")
    depth_reduction_pct = 0
    if optimization_info['initial_depth'] > 0:
        depth_reduction_pct = optimization_info['depth_reduction'] / optimization_info['initial_depth'] * 100
    print(f"  Depth reduction: {optimization_info['depth_reduction']} ({depth_reduction_pct:.2f}%)")
    
    print(f"  Original gate count: {optimization_info['initial_gate_count']}")
    print(f"  Optimized gate count: {optimization_info['optimized_gate_count']}")
    gate_reduction_pct = 0
    if optimization_info['initial_gate_count'] > 0:
        gate_reduction_pct = optimization_info['gate_count_reduction'] / optimization_info['initial_gate_count'] * 100
    print(f"  Gate count reduction: {optimization_info['gate_count_reduction']} ({gate_reduction_pct:.2f}%)")
    
    # Check circuit equivalence
    metrics = CircuitMetrics(cost_database)
    equivalent = metrics.check_equivalence(circuit, optimized_circuit.circuit)
    print(f"Circuits are equivalent: {equivalent}")
    
    # Print the optimized circuit
    print(f"Optimized {name} circuit:")
    print(optimized_circuit.circuit)
    
    return optimized_circuit, agent

def main():
    """Main function."""
    # Create different types of circuits to benchmark
    print("Creating benchmark circuits...")
    
    # 1. QFT circuit with 4 qubits
    qft_circuit = QFT(4)
    
    # 2. Random circuit with 4 qubits and 10 gates
    random_circ = random_circuit(4, 10, seed=42)
    
    # 3. Parameterized circuit (Real Amplitudes)
    real_amp_circuit = RealAmplitudes(4, reps=1)
    real_amp_circuit = real_amp_circuit.bind_parameters([0.1] * len(real_amp_circuit.parameters))
    
    # 4. Efficient SU2 circuit
    su2_circuit = EfficientSU2(4, reps=1)
    su2_circuit = su2_circuit.bind_parameters([0.1] * len(su2_circuit.parameters))
    
    # Train a single agent on the random circuit
    _, agent = optimize_circuit(random_circ, "Random", steps=500)
    
    # Use the same agent to optimize other circuits
    optimize_circuit(qft_circuit, "QFT", agent=agent, steps=0)
    optimize_circuit(real_amp_circuit, "RealAmplitudes", agent=agent, steps=0)
    optimize_circuit(su2_circuit, "EfficientSU2", agent=agent, steps=0)
    
    print("\nBenchmarking complete!")

if __name__ == '__main__':
    main()
