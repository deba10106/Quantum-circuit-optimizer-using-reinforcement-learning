"""
Cost Database
===========

This module provides a database of gate costs and error rates for quantum computers.
"""

import json
import os
import numpy as np
import pandas as pd

class CostDatabase:
    """
    A database of gate costs and error rates for quantum computers.
    """
    
    def __init__(self, backend_name=None, load_from_file=None):
        """
        Initialize the cost database.
        
        Args:
            backend_name (str, optional): The name of the backend to load data for.
            load_from_file (str, optional): Path to a JSON file to load data from.
        """
        self.gate_costs = {}
        self.error_rates = {}
        self.backend_name = backend_name
        
        if load_from_file:
            self.load_from_file(load_from_file)
        elif backend_name:
            print(f"Note: Direct backend loading is not available. Using default data instead.")
            self._init_default_data()
        else:
            self._init_default_data()
    
    def _init_default_data(self):
        """Initialize with default data based on typical IBM quantum computers."""
        # Default gate costs (execution time in ns)
        self.gate_costs = {
            'id': 35.5,
            'x': 35.5,
            'sx': 35.5,
            'rz': 0,  # Virtual gate, no real execution time
            'cx': 300.0,
            'ecr': 300.0,
            'reset': 1000.0,
            'measure': 1000.0
        }
        
        # Default error rates
        self.error_rates = {
            'id': 0.0001,
            'x': 0.0005,
            'sx': 0.0005,
            'rz': 0.0001,
            'cx': 0.01,
            'ecr': 0.01,
            'reset': 0.02,
            'measure': 0.02
        }
    
    def load_from_file(self, filepath):
        """
        Load gate costs and error rates from a JSON file.
        
        Args:
            filepath (str): Path to the JSON file.
        """
        if not os.path.exists(filepath):
            print(f"File {filepath} not found. Using default data.")
            self._init_default_data()
            return
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.gate_costs = data.get('gate_costs', {})
            self.error_rates = data.get('error_rates', {})
            self.backend_name = data.get('backend_name', None)
            
            print(f"Loaded data from {filepath}")
        except Exception as e:
            print(f"Error loading data from {filepath}: {e}")
            self._init_default_data()
    
    def save_to_file(self, filepath):
        """
        Save gate costs and error rates to a JSON file.
        
        Args:
            filepath (str): Path to the JSON file.
        """
        data = {
            'gate_costs': self.gate_costs,
            'error_rates': self.error_rates,
            'backend_name': self.backend_name
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"Saved data to {filepath}")
        except Exception as e:
            print(f"Error saving data to {filepath}: {e}")
    
    def get_gate_cost(self, gate_name):
        """
        Get the cost of a gate.
        
        Args:
            gate_name (str): The name of the gate.
            
        Returns:
            float: The cost of the gate.
        """
        return self.gate_costs.get(gate_name, 0.0)
    
    def get_error_rate(self, gate_name):
        """
        Get the error rate of a gate.
        
        Args:
            gate_name (str): The name of the gate.
            
        Returns:
            float: The error rate of the gate.
        """
        return self.error_rates.get(gate_name, 0.0)
    
    def calculate_circuit_cost(self, circuit):
        """
        Calculate the total cost of a circuit.
        
        Args:
            circuit: A quantum circuit.
            
        Returns:
            float: The total cost of the circuit.
        """
        total_cost = 0.0
        
        for gate_name, count in circuit.count_ops().items():
            gate_cost = self.get_gate_cost(gate_name)
            total_cost += gate_cost * count
        
        return total_cost
    
    def calculate_circuit_error(self, circuit):
        """
        Calculate the error probability of a circuit.
        
        Args:
            circuit: A quantum circuit.
            
        Returns:
            float: The error probability of the circuit.
        """
        error_prob = 1.0
        
        for gate_name, count in circuit.count_ops().items():
            gate_error = self.get_error_rate(gate_name)
            # Calculate probability of no error for this gate type
            no_error_prob = (1 - gate_error) ** count
            # Multiply by probability of no error
            error_prob *= no_error_prob
        
        # Return probability of at least one error
        return 1 - error_prob
