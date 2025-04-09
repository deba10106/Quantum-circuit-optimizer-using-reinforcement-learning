"""
Example: CUDA-Accelerated Quantum Circuit Optimization
====================================================

This example demonstrates how to use CUDA acceleration for the quantum circuit optimizer
if a compatible GPU is available.
"""

import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.visualization import plot_histogram
from qiskit import Aer, execute

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from quantum_circuit_optimizer.quantum_circuit.circuit import QuantumCircuit as QCO
from quantum_circuit_optimizer.cost_database.cost_database import CostDatabase
from quantum_circuit_optimizer.environment.quantum_environment import QuantumEnvironment
from quantum_circuit_optimizer.agent.dqn_agent import DQNAgent
from quantum_circuit_optimizer.agent.ppo_agent import PPOAgent
from quantum_circuit_optimizer.evaluation.metrics import CircuitMetrics
from quantum_circuit_optimizer.evaluation.benchmarks import CircuitBenchmarks
from quantum_circuit_optimizer.evaluation.visualization import CircuitVisualizer

def check_cuda_availability():
    """Check if CUDA is available and print device information."""
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices: {device_count}")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            device_capability = torch.cuda.get_device_capability(i)
            print(f"  Device {i}: {device_name} (Compute Capability {device_capability[0]}.{device_capability[1]})")
        
        # Get current device
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device: {current_device} ({torch.cuda.get_device_name(current_device)})")
        
        # Memory information
        print(f"Total memory: {torch.cuda.get_device_properties(current_device).total_memory / 1024**3:.2f} GB")
        print(f"Allocated memory: {torch.cuda.memory_allocated(current_device) / 1024**3:.2f} GB")
        print(f"Cached memory: {torch.cuda.memory_reserved(current_device) / 1024**3:.2f} GB")
    
    return cuda_available

def benchmark_device_performance(device_name='auto'):
    """Benchmark the performance of CPU vs. GPU for neural network operations."""
    print(f"\nBenchmarking device performance for '{device_name}'...")
    
    # Determine device
    if device_name == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_name)
    
    print(f"Using device: {device}")
    
    # Create a simple neural network
    class SimpleNN(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(SimpleNN, self).__init__()
            self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
            self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = torch.nn.Linear(hidden_dim, output_dim)
            self.relu = torch.nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # Create model
    input_dim = 128
    hidden_dim = 256
    output_dim = 64
    model = SimpleNN(input_dim, hidden_dim, output_dim).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create data
    batch_size = 1024
    x = torch.randn(batch_size, input_dim).to(device)
    y = torch.randn(batch_size, output_dim).to(device)
    
    # Benchmark forward and backward pass
    num_iterations = 1000
    
    # Warmup
    for _ in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()
        optimizer.step()
    
    # Benchmark
    start_time = time.time()
    
    for _ in range(num_iterations):
        optimizer.zero_grad()
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()
        optimizer.step()
    
    end_time = time.time()
    
    # Calculate performance
    elapsed_time = end_time - start_time
    iterations_per_second = num_iterations / elapsed_time
    
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    print(f"Iterations per second: {iterations_per_second:.2f}")
    
    return device, iterations_per_second

def optimize_circuit_with_device(device_name='auto'):
    """Optimize a quantum circuit using the specified device."""
    # Create a cost database
    cost_database = CostDatabase()
    
    # Create a quantum circuit (QFT on 5 qubits)
    print("\nCreating a QFT circuit with 5 qubits...")
    circuit = QFT(5)
    
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
    
    # Create a DQN agent with the specified device
    print(f"Creating the DQN agent with device '{device_name}'...")
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
        device=device_name
    )
    
    # Train the agent
    print("Training the agent for 1000 steps...")
    start_time = time.time()
    train_results = agent.train(env, 1000)
    train_time = time.time() - start_time
    
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Optimize the circuit
    print("Optimizing the circuit...")
    start_time = time.time()
    optimized_circuit, optimization_info = agent.optimize_circuit(
        env, circuit, max_steps=50, render=False
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
    
    return train_time, optimization_time, optimization_info

def compare_devices():
    """Compare performance between CPU and GPU."""
    print("\n=== Device Performance Comparison ===")
    
    # Benchmark CPU
    cpu_device, cpu_perf = benchmark_device_performance('cpu')
    
    # Benchmark GPU if available
    if torch.cuda.is_available():
        gpu_device, gpu_perf = benchmark_device_performance('cuda')
        
        # Calculate speedup
        speedup = gpu_perf / cpu_perf
        print(f"\nGPU is {speedup:.2f}x faster than CPU for neural network operations")
    
    # Optimize circuit with CPU
    print("\n=== Circuit Optimization with CPU ===")
    cpu_train_time, cpu_opt_time, cpu_results = optimize_circuit_with_device('cpu')
    
    # Optimize circuit with GPU if available
    if torch.cuda.is_available():
        print("\n=== Circuit Optimization with GPU ===")
        gpu_train_time, gpu_opt_time, gpu_results = optimize_circuit_with_device('cuda')
        
        # Calculate speedup
        train_speedup = cpu_train_time / gpu_train_time
        opt_speedup = cpu_opt_time / gpu_opt_time
        
        print("\n=== Performance Comparison ===")
        print(f"Training time: CPU {cpu_train_time:.2f}s vs GPU {gpu_train_time:.2f}s (Speedup: {train_speedup:.2f}x)")
        print(f"Optimization time: CPU {cpu_opt_time:.2f}s vs GPU {gpu_opt_time:.2f}s (Speedup: {opt_speedup:.2f}x)")
        
        # Compare optimization results
        print("\n=== Optimization Results Comparison ===")
        print(f"CPU depth reduction: {cpu_results['depth_reduction']} ({cpu_results['depth_reduction'] / cpu_results['initial_depth'] * 100:.2f}%)")
        print(f"GPU depth reduction: {gpu_results['depth_reduction']} ({gpu_results['depth_reduction'] / gpu_results['initial_depth'] * 100:.2f}%)")
        
        print(f"CPU gate count reduction: {cpu_results['gate_count_reduction']} ({cpu_results['gate_count_reduction'] / cpu_results['initial_gate_count'] * 100:.2f}%)")
        print(f"GPU gate count reduction: {gpu_results['gate_count_reduction']} ({gpu_results['gate_count_reduction'] / gpu_results['initial_gate_count'] * 100:.2f}%)")
        
        # Visualize performance comparison
        labels = ['Training Time', 'Optimization Time']
        cpu_times = [cpu_train_time, cpu_opt_time]
        gpu_times = [gpu_train_time, gpu_opt_time]
        
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, cpu_times, width, label='CPU')
        rects2 = ax.bar(x + width/2, gpu_times, width, label='GPU')
        
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Performance Comparison: CPU vs GPU')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        # Add labels on top of bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}s',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function."""
    print("=== CUDA Integration for Quantum Circuit Optimizer ===")
    
    # Check CUDA availability
    cuda_available = check_cuda_availability()
    
    if cuda_available:
        # Compare CPU and GPU performance
        compare_devices()
    else:
        print("\nCUDA is not available. Running on CPU only.")
        
        # Benchmark CPU performance
        benchmark_device_performance('cpu')
        
        # Optimize circuit with CPU
        optimize_circuit_with_device('cpu')
    
    print("\nDone!")

if __name__ == '__main__':
    main()
