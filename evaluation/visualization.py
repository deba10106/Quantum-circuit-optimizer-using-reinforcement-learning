"""
Circuit Visualization
==================

This module provides visualization tools for quantum circuit optimizations.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from qiskit.visualization import plot_circuit_layout, plot_gate_map, plot_histogram
from qiskit.visualization import plot_bloch_multivector, plot_state_city

class CircuitVisualizer:
    """
    Provides visualization tools for quantum circuit optimizations.
    """
    
    def __init__(self, figsize=(12, 8), style='default'):
        """
        Initialize the circuit visualizer.
        
        Args:
            figsize (tuple): Figure size for plots.
            style (str): Matplotlib style.
        """
        self.figsize = figsize
        
        # Set plot style
        if style == 'default':
            plt.style.use('default')
        else:
            plt.style.use(style)
    
    def plot_circuit(self, circuit, title=None, filename=None):
        """
        Plot a quantum circuit.
        
        Args:
            circuit: The circuit to plot.
            title (str): Title for the plot.
            filename (str): Filename to save the plot to.
            
        Returns:
            The circuit diagram.
        """
        # Get the Qiskit circuit
        if hasattr(circuit, 'circuit'):
            qiskit_circuit = circuit.circuit
        else:
            qiskit_circuit = circuit
        
        # Create the plot
        fig = plt.figure(figsize=self.figsize)
        
        # Draw the circuit
        circuit_diagram = qiskit_circuit.draw(output='mpl', style={'backgroundcolor': '#FFFFFF'})
        
        # Set title
        if title:
            plt.title(title, fontsize=16)
        
        # Save to file if specified
        if filename:
            plt.savefig(filename, bbox_inches='tight', dpi=300)
        
        plt.close(fig)
        
        return circuit_diagram
    
    def plot_circuit_comparison(self, original_circuit, optimized_circuit, title=None, filename=None):
        """
        Plot a comparison of original and optimized circuits.
        
        Args:
            original_circuit: The original circuit.
            optimized_circuit: The optimized circuit.
            title (str): Title for the plot.
            filename (str): Filename to save the plot to.
            
        Returns:
            The figure.
        """
        # Get the Qiskit circuits
        if hasattr(original_circuit, 'circuit'):
            original_qiskit_circuit = original_circuit.circuit
        else:
            original_qiskit_circuit = original_circuit
            
        if hasattr(optimized_circuit, 'circuit'):
            optimized_qiskit_circuit = optimized_circuit.circuit
        else:
            optimized_qiskit_circuit = optimized_circuit
        
        # Create the plot
        fig, axes = plt.subplots(2, 1, figsize=self.figsize)
        
        # Draw the original circuit
        original_qiskit_circuit.draw(output='mpl', ax=axes[0], style={'backgroundcolor': '#FFFFFF'})
        axes[0].set_title('Original Circuit', fontsize=14)
        
        # Draw the optimized circuit
        optimized_qiskit_circuit.draw(output='mpl', ax=axes[1], style={'backgroundcolor': '#FFFFFF'})
        axes[1].set_title('Optimized Circuit', fontsize=14)
        
        # Set title
        if title:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        
        # Save to file if specified
        if filename:
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close(fig)
        
        return fig
    
    def plot_optimization_metrics(self, metrics, title=None, filename=None):
        """
        Plot optimization metrics.
        
        Args:
            metrics (dict): Dictionary of metrics.
            title (str): Title for the plot.
            filename (str): Filename to save the plot to.
            
        Returns:
            The figure.
        """
        # Create the plot
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Plot depth comparison
        axes[0, 0].bar(['Original', 'Optimized'], [metrics['original_depth'], metrics['depth']])
        axes[0, 0].set_title('Circuit Depth', fontsize=12)
        axes[0, 0].set_ylabel('Depth')
        
        # Plot gate count comparison
        axes[0, 1].bar(['Original', 'Optimized'], [metrics['original_gate_count'], metrics['gate_count']])
        axes[0, 1].set_title('Gate Count', fontsize=12)
        axes[0, 1].set_ylabel('Count')
        
        # Plot two-qubit gate count comparison
        axes[1, 0].bar(['Original', 'Optimized'], [metrics['original_two_qubit_gate_count'], metrics['two_qubit_gate_count']])
        axes[1, 0].set_title('Two-Qubit Gate Count', fontsize=12)
        axes[1, 0].set_ylabel('Count')
        
        # Plot cost comparison if available
        if 'original_cost' in metrics and 'cost' in metrics:
            axes[1, 1].bar(['Original', 'Optimized'], [metrics['original_cost'], metrics['cost']])
            axes[1, 1].set_title('Circuit Cost', fontsize=12)
            axes[1, 1].set_ylabel('Cost')
        else:
            axes[1, 1].axis('off')
        
        # Set title
        if title:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        
        # Save to file if specified
        if filename:
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close(fig)
        
        return fig
    
    def plot_gate_distribution(self, circuit, title=None, filename=None):
        """
        Plot the distribution of gates in a circuit.
        
        Args:
            circuit: The circuit to plot.
            title (str): Title for the plot.
            filename (str): Filename to save the plot to.
            
        Returns:
            The figure.
        """
        # Get the Qiskit circuit
        if hasattr(circuit, 'circuit'):
            qiskit_circuit = circuit.circuit
        else:
            qiskit_circuit = circuit
        
        # Count gates
        gate_counts = {}
        for op in qiskit_circuit.data:
            gate_name = op[0].name
            if gate_name not in ['barrier', 'snapshot']:
                gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
        
        # Create the plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Sort gates by count
        sorted_gates = sorted(gate_counts.items(), key=lambda x: x[1], reverse=True)
        gates = [g[0] for g in sorted_gates]
        counts = [g[1] for g in sorted_gates]
        
        # Plot the distribution
        ax.bar(gates, counts)
        ax.set_xlabel('Gate Type', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        
        # Set title
        if title:
            ax.set_title(title, fontsize=16)
        else:
            ax.set_title('Gate Distribution', fontsize=16)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save to file if specified
        if filename:
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close(fig)
        
        return fig
    
    def plot_optimization_progress(self, rewards, metrics, title=None, filename=None):
        """
        Plot the progress of optimization.
        
        Args:
            rewards (list): List of rewards during optimization.
            metrics (list): List of metrics during optimization.
            title (str): Title for the plot.
            filename (str): Filename to save the plot to.
            
        Returns:
            The figure.
        """
        # Create the plot
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Plot rewards
        axes[0, 0].plot(rewards)
        axes[0, 0].set_title('Rewards', fontsize=12)
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Reward')
        
        # Plot depth
        depths = [m['depth'] for m in metrics]
        axes[0, 1].plot(depths)
        axes[0, 1].set_title('Circuit Depth', fontsize=12)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Depth')
        
        # Plot gate count
        gate_counts = [m['gate_count'] for m in metrics]
        axes[1, 0].plot(gate_counts)
        axes[1, 0].set_title('Gate Count', fontsize=12)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Count')
        
        # Plot cost if available
        if 'cost' in metrics[0]:
            costs = [m['cost'] for m in metrics]
            axes[1, 1].plot(costs)
            axes[1, 1].set_title('Circuit Cost', fontsize=12)
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Cost')
        else:
            axes[1, 1].axis('off')
        
        # Set title
        if title:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        
        # Save to file if specified
        if filename:
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close(fig)
        
        return fig
    
    def plot_benchmark_results(self, results, title=None, filename=None):
        """
        Plot benchmark results.
        
        Args:
            results (dict): Dictionary of benchmark results.
            title (str): Title for the plot.
            filename (str): Filename to save the plot to.
            
        Returns:
            The figure.
        """
        # Create the plot
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Extract data
        depth_reductions = [m['depth_reduction'] for m in results['optimized_metrics']]
        gate_count_reductions = [m['gate_count_reduction'] for m in results['optimized_metrics']]
        optimization_times = results['optimization_times']
        equivalence_checks = results['equivalence_checks']
        
        # Plot depth reductions
        axes[0, 0].bar(range(len(depth_reductions)), depth_reductions)
        axes[0, 0].set_title('Depth Reduction', fontsize=12)
        axes[0, 0].set_xlabel('Circuit')
        axes[0, 0].set_ylabel('Reduction')
        
        # Plot gate count reductions
        axes[0, 1].bar(range(len(gate_count_reductions)), gate_count_reductions)
        axes[0, 1].set_title('Gate Count Reduction', fontsize=12)
        axes[0, 1].set_xlabel('Circuit')
        axes[0, 1].set_ylabel('Reduction')
        
        # Plot optimization times
        axes[1, 0].bar(range(len(optimization_times)), optimization_times)
        axes[1, 0].set_title('Optimization Time', fontsize=12)
        axes[1, 0].set_xlabel('Circuit')
        axes[1, 0].set_ylabel('Time (s)')
        
        # Plot equivalence checks
        axes[1, 1].bar(['Equivalent', 'Not Equivalent'], 
                      [sum(equivalence_checks), len(equivalence_checks) - sum(equivalence_checks)])
        axes[1, 1].set_title('Equivalence Checks', fontsize=12)
        axes[1, 1].set_ylabel('Count')
        
        # Set title
        if title:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        
        # Save to file if specified
        if filename:
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close(fig)
        
        return fig
    
    def plot_optimizer_comparison(self, comparison, metric='mean_depth_reduction_percent', title=None, filename=None):
        """
        Plot a comparison of optimizers.
        
        Args:
            comparison (dict): Dictionary of comparison results.
            metric (str): Metric to compare.
            title (str): Title for the plot.
            filename (str): Filename to save the plot to.
            
        Returns:
            The figure.
        """
        # Create the plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Extract data
        optimizers = list(comparison.keys())
        values = [comparison[opt]['summary'][metric] for opt in optimizers]
        
        # Plot the comparison
        ax.bar(optimizers, values)
        ax.set_xlabel('Optimizer', fontsize=12)
        
        # Set y-label based on metric
        if 'percent' in metric:
            ax.set_ylabel('Percentage (%)', fontsize=12)
        elif 'time' in metric:
            ax.set_ylabel('Time (s)', fontsize=12)
        elif 'rate' in metric:
            ax.set_ylabel('Rate', fontsize=12)
        else:
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        
        # Set title
        if title:
            ax.set_title(title, fontsize=16)
        else:
            ax.set_title(f'Comparison of {metric.replace("_", " ").title()}', fontsize=16)
        
        plt.tight_layout()
        
        # Save to file if specified
        if filename:
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close(fig)
        
        return fig
    
    def plot_state_visualization(self, circuit, backend=None, title=None, filename=None):
        """
        Plot a visualization of the quantum state produced by a circuit.
        
        Args:
            circuit: The circuit to visualize.
            backend: The backend to use for simulation.
            title (str): Title for the plot.
            filename (str): Filename to save the plot to.
            
        Returns:
            The figure.
        """
        from qiskit import Aer, execute
        from qiskit.quantum_info import Statevector
        
        # Get the Qiskit circuit
        if hasattr(circuit, 'circuit'):
            qiskit_circuit = circuit.circuit
        else:
            qiskit_circuit = circuit
        
        # If no backend is provided, use a statevector simulator
        if backend is None:
            backend = Aer.get_backend('statevector_simulator')
        
        # Execute the circuit
        result = execute(qiskit_circuit, backend).result()
        
        # Get the statevector
        if backend.name() == 'statevector_simulator':
            statevector = result.get_statevector(qiskit_circuit)
        else:
            # If not using statevector simulator, create a statevector from the circuit
            statevector = Statevector.from_instruction(qiskit_circuit)
        
        # Create the plot
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Plot the statevector as a city plot
        plot_state_city(statevector, title="State City Plot", ax=axes[0])
        
        # Plot the Bloch sphere representation
        if qiskit_circuit.num_qubits <= 5:  # Bloch sphere visualization is practical for small number of qubits
            plot_bloch_multivector(statevector, title="Bloch Sphere", ax=axes[1])
        else:
            axes[1].text(0.5, 0.5, "Bloch sphere visualization not shown\n(too many qubits)", 
                        horizontalalignment='center', verticalalignment='center')
            axes[1].axis('off')
        
        # Set title
        if title:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        
        # Save to file if specified
        if filename:
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close(fig)
        
        return fig
