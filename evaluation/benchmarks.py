"""
Circuit Benchmarks
===============

This module provides benchmarks for evaluating quantum circuit optimizations.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import (
    QFT, HGate, CXGate, TGate, ZGate, XGate, RZGate, RXGate, RYGate,
    CZGate, SwapGate, CCXGate, U1Gate, U2Gate, U3Gate
)

from .metrics import CircuitMetrics

class CircuitBenchmarks:
    """
    Provides benchmark circuits and evaluation for quantum circuit optimizations.
    """
    
    def __init__(self, cost_database=None):
        """
        Initialize the circuit benchmarks.
        
        Args:
            cost_database: A CostDatabase instance for calculating costs and errors.
        """
        self.cost_database = cost_database
        self.metrics = CircuitMetrics(cost_database)
    
    def generate_random_circuits(self, num_circuits=10, num_qubits=5, min_depth=5, max_depth=20, seed=None):
        """
        Generate a set of random quantum circuits.
        
        Args:
            num_circuits (int): Number of circuits to generate.
            num_qubits (int): Number of qubits in each circuit.
            min_depth (int): Minimum circuit depth.
            max_depth (int): Maximum circuit depth.
            seed (int): Random seed for reproducibility.
            
        Returns:
            list: List of random quantum circuits.
        """
        circuits = []
        
        for i in range(num_circuits):
            # Randomly select a depth
            depth = np.random.randint(min_depth, max_depth + 1)
            
            # Generate a random circuit
            circuit = random_circuit(num_qubits, depth, 
                                    max_operands=2, 
                                    measure=False, 
                                    seed=seed + i if seed is not None else None)
            
            circuits.append(circuit)
        
        return circuits
    
    def generate_qft_circuits(self, min_qubits=3, max_qubits=10):
        """
        Generate Quantum Fourier Transform benchmark circuits.
        
        Args:
            min_qubits (int): Minimum number of qubits.
            max_qubits (int): Maximum number of qubits.
            
        Returns:
            list: A list of QFT circuits.
        """
        circuits = []
        for num_qubits in range(min_qubits, max_qubits + 1):
            # Create a QFT circuit
            circuit = QFT(num_qubits)
            
            circuits.append(circuit)
        
        return circuits
    
    def generate_adder_circuits(self, min_bits=2, max_bits=5):
        """
        Generate quantum adder benchmark circuits.
        
        Args:
            min_bits (int): Minimum number of bits.
            max_bits (int): Maximum number of bits.
            
        Returns:
            list: A list of adder circuits.
        """
        circuits = []
        for num_bits in range(min_bits, max_bits + 1):
            # Create a quantum adder circuit
            circuit = self._create_quantum_adder(num_bits)
            
            circuits.append(circuit)
        
        return circuits
    
    def _create_quantum_adder(self, num_bits):
        """
        Create a quantum adder circuit.
        
        Args:
            num_bits (int): Number of bits.
            
        Returns:
            QuantumCircuit: A quantum adder circuit.
        """
        # Create a quantum circuit for adding two n-bit numbers
        # This is a simplified ripple-carry adder
        
        # We need 2*num_bits qubits for the two numbers
        # and num_bits+1 qubits for the carry and result
        total_qubits = 3 * num_bits + 1
        
        circuit = QuantumCircuit(total_qubits)
        
        # Initialize the first register (first number)
        for i in range(num_bits):
            # Randomly initialize to 0 or 1
            if np.random.random() > 0.5:
                circuit.x(i)
        
        # Initialize the second register (second number)
        for i in range(num_bits):
            # Randomly initialize to 0 or 1
            if np.random.random() > 0.5:
                circuit.x(num_bits + i)
        
        # Implement the ripple-carry adder
        for i in range(num_bits):
            # Apply CNOT gates
            circuit.cx(i, 2 * num_bits + i)
            circuit.cx(num_bits + i, 2 * num_bits + i)
            
            # Apply Toffoli gates for carry
            if i < num_bits - 1:
                circuit.ccx(i, num_bits + i, 2 * num_bits + i + 1)
                
                # Apply additional gates for carry propagation
                circuit.cx(i, num_bits + i)
                circuit.ccx(2 * num_bits + i, num_bits + i, 2 * num_bits + i + 1)
                circuit.cx(i, num_bits + i)
        
        return circuit
    
    def generate_grover_circuits(self, min_qubits=2, max_qubits=5):
        """
        Generate Grover's algorithm benchmark circuits.
        
        Args:
            min_qubits (int): Minimum number of qubits.
            max_qubits (int): Maximum number of qubits.
            
        Returns:
            list: A list of Grover's algorithm circuits.
        """
        circuits = []
        for num_qubits in range(min_qubits, max_qubits + 1):
            # Create a Grover's algorithm circuit
            circuit = self._create_grover_circuit(num_qubits)
            
            circuits.append(circuit)
        
        return circuits
    
    def _create_grover_circuit(self, num_qubits):
        """
        Create a Grover's algorithm circuit.
        
        Args:
            num_qubits (int): Number of qubits.
            
        Returns:
            QuantumCircuit: A Grover's algorithm circuit.
        """
        # Create a quantum circuit for Grover's algorithm
        # This is a simplified version with a random oracle
        
        circuit = QuantumCircuit(num_qubits)
        
        # Apply Hadamard gates to all qubits
        for i in range(num_qubits):
            circuit.h(i)
        
        # Number of iterations (approximately sqrt(N))
        num_iterations = int(np.sqrt(2**num_qubits))
        
        # Randomly choose a marked state
        marked_state = np.random.randint(0, 2**num_qubits)
        marked_state_bin = format(marked_state, f'0{num_qubits}b')
        
        # Perform Grover iterations
        for _ in range(num_iterations):
            # Oracle: Flip the phase of the marked state
            # For simplicity, we'll use a circuit that marks a random state
            
            # Apply X gates to qubits where the marked state has a 0
            for i in range(num_qubits):
                if marked_state_bin[i] == '0':
                    circuit.x(i)
            
            # Apply multi-controlled Z gate
            if num_qubits > 1:
                # For simplicity, we'll use a series of controlled-Z gates
                for i in range(num_qubits - 1):
                    circuit.cz(i, i + 1)
            else:
                circuit.z(0)
            
            # Apply X gates again to restore the state
            for i in range(num_qubits):
                if marked_state_bin[i] == '0':
                    circuit.x(i)
            
            # Diffusion operator
            # Apply Hadamard gates to all qubits
            for i in range(num_qubits):
                circuit.h(i)
            
            # Apply X gates to all qubits
            for i in range(num_qubits):
                circuit.x(i)
            
            # Apply multi-controlled Z gate
            if num_qubits > 1:
                # For simplicity, we'll use a series of controlled-Z gates
                for i in range(num_qubits - 1):
                    circuit.cz(i, i + 1)
            else:
                circuit.z(0)
            
            # Apply X gates to all qubits
            for i in range(num_qubits):
                circuit.x(i)
            
            # Apply Hadamard gates to all qubits
            for i in range(num_qubits):
                circuit.h(i)
        
        return circuit
    
    def evaluate_optimizer(self, optimizer, circuits, verbose=True):
        """
        Evaluate an optimizer on benchmark circuits.
        
        Args:
            optimizer: The optimizer to evaluate.
            circuits (list): List of circuits to optimize.
            verbose (bool): Whether to print progress.
            
        Returns:
            dict: Evaluation results.
        """
        results = {
            'original_metrics': [],
            'optimized_metrics': [],
            'optimization_times': [],
            'equivalence_checks': []
        }
        
        for i, circuit in enumerate(circuits):
            if verbose:
                print(f"Optimizing circuit {i+1}/{len(circuits)}...")
            
            # Calculate original metrics
            original_metrics = self.metrics.calculate_metrics(circuit)
            results['original_metrics'].append(original_metrics)
            
            # Optimize the circuit
            start_time = time.time()
            optimized_circuit = optimizer.optimize(circuit)
            optimization_time = time.time() - start_time
            
            # Calculate optimized metrics
            optimized_metrics = self.metrics.calculate_metrics(optimized_circuit, circuit)
            results['optimized_metrics'].append(optimized_metrics)
            
            # Record optimization time
            results['optimization_times'].append(optimization_time)
            
            # Check equivalence
            equivalence = self.metrics.check_equivalence(circuit, optimized_circuit)
            results['equivalence_checks'].append(equivalence)
            
            if verbose:
                print(f"  Original depth: {original_metrics['depth']}")
                print(f"  Optimized depth: {optimized_metrics['depth']}")
                print(f"  Depth reduction: {optimized_metrics['depth_reduction']} ({optimized_metrics['depth_reduction_percent']:.2f}%)")
                print(f"  Gate count reduction: {optimized_metrics['gate_count_reduction']} ({optimized_metrics['gate_count_reduction_percent']:.2f}%)")
                print(f"  Optimization time: {optimization_time:.4f} seconds")
                print(f"  Equivalent: {equivalence}")
                print()
        
        # Calculate summary statistics
        results['summary'] = self._calculate_summary_statistics(results)
        
        if verbose:
            print("Summary:")
            print(f"  Mean depth reduction: {results['summary']['mean_depth_reduction']:.2f} ({results['summary']['mean_depth_reduction_percent']:.2f}%)")
            print(f"  Mean gate count reduction: {results['summary']['mean_gate_count_reduction']:.2f} ({results['summary']['mean_gate_count_reduction_percent']:.2f}%)")
            print(f"  Mean optimization time: {results['summary']['mean_optimization_time']:.4f} seconds")
            print(f"  Equivalence rate: {results['summary']['equivalence_rate'] * 100:.2f}%")
        
        return results
    
    def _calculate_summary_statistics(self, results):
        """
        Calculate summary statistics from evaluation results.
        
        Args:
            results (dict): Evaluation results.
            
        Returns:
            dict: Summary statistics.
        """
        summary = {}
        
        # Extract metrics
        depth_reductions = [metrics['depth_reduction'] for metrics in results['optimized_metrics']]
        depth_reduction_percents = [metrics['depth_reduction_percent'] for metrics in results['optimized_metrics']]
        gate_count_reductions = [metrics['gate_count_reduction'] for metrics in results['optimized_metrics']]
        gate_count_reduction_percents = [metrics['gate_count_reduction_percent'] for metrics in results['optimized_metrics']]
        
        # Calculate statistics
        summary['mean_depth_reduction'] = np.mean(depth_reductions)
        summary['std_depth_reduction'] = np.std(depth_reductions)
        summary['mean_depth_reduction_percent'] = np.mean(depth_reduction_percents)
        
        summary['mean_gate_count_reduction'] = np.mean(gate_count_reductions)
        summary['std_gate_count_reduction'] = np.std(gate_count_reductions)
        summary['mean_gate_count_reduction_percent'] = np.mean(gate_count_reduction_percents)
        
        summary['mean_optimization_time'] = np.mean(results['optimization_times'])
        summary['std_optimization_time'] = np.std(results['optimization_times'])
        
        summary['equivalence_rate'] = np.mean(results['equivalence_checks'])
        
        return summary
    
    def compare_optimizers(self, optimizers, circuits, verbose=True):
        """
        Compare multiple optimizers on benchmark circuits.
        
        Args:
            optimizers (dict): Dictionary of optimizers to compare.
            circuits (list): List of circuits to optimize.
            verbose (bool): Whether to print progress.
            
        Returns:
            dict: Comparison results.
        """
        comparison = {}
        
        for name, optimizer in optimizers.items():
            if verbose:
                print(f"Evaluating optimizer: {name}")
            
            results = self.evaluate_optimizer(optimizer, circuits, verbose=verbose)
            comparison[name] = results
            
            if verbose:
                print()
        
        # Calculate comparison summary
        if verbose:
            print("Comparison Summary:")
            for name, results in comparison.items():
                print(f"  {name}:")
                print(f"    Mean depth reduction: {results['summary']['mean_depth_reduction']:.2f} ({results['summary']['mean_depth_reduction_percent']:.2f}%)")
                print(f"    Mean gate count reduction: {results['summary']['mean_gate_count_reduction']:.2f} ({results['summary']['mean_gate_count_reduction_percent']:.2f}%)")
                print(f"    Mean optimization time: {results['summary']['mean_optimization_time']:.4f} seconds")
                print(f"    Equivalence rate: {results['summary']['equivalence_rate'] * 100:.2f}%")
                print()
        
        return comparison
