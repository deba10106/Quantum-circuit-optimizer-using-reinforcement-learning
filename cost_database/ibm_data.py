"""
IBM Quantum Data
==============

This module provides specific data about IBM quantum computers,
including gate costs and error rates.
"""

import json
import os
import numpy as np
import pandas as pd

class IBMQuantumData:
    """
    A class for accessing and managing IBM quantum computer data.
    """
    
    def __init__(self):
        """Initialize with default IBM quantum data."""
        self.backends = {}
        self._init_default_data()
    
    def _init_default_data(self):
        """Initialize with default data for common IBM quantum backends."""
        # IBM Quantum Falcon Processors - Default data based on published information
        # These are approximate values and should be updated with real data when available
        
        # IBM Quantum Falcon r5.11 (127 qubit) - 'ibm_sherbrooke'
        sherbrooke = {
            'gate_costs': {
                'id': 35.5,
                'x': 35.5,
                'sx': 35.5,
                'rz': 0,  # Virtual gate
                'ecr': 300.0,
                'measure': 1000.0,
                'reset': 1000.0
            },
            'error_rates': {
                'id': 0.0001,
                'x': 0.0003,
                'sx': 0.0003,
                'rz': 0.0001,  # Virtual gate
                'ecr': 0.007,  # Typical two-qubit gate error for Falcon r5
                'measure': 0.015,
                'reset': 0.015
            },
            'basis_gates': ['id', 'rz', 'sx', 'x', 'ecr', 'reset', 'measure'],
            'coupling_map': self._generate_heavy_hex_coupling(127)
        }
        
        # IBM Quantum Falcon r4 (27 qubit) - 'ibm_cairo'
        cairo = {
            'gate_costs': {
                'id': 35.5,
                'x': 35.5,
                'sx': 35.5,
                'rz': 0,  # Virtual gate
                'cx': 300.0,
                'measure': 1000.0,
                'reset': 1000.0
            },
            'error_rates': {
                'id': 0.0001,
                'x': 0.0004,
                'sx': 0.0004,
                'rz': 0.0001,  # Virtual gate
                'cx': 0.01,  # Typical two-qubit gate error for Falcon r4
                'measure': 0.02,
                'reset': 0.02
            },
            'basis_gates': ['id', 'rz', 'sx', 'x', 'cx', 'reset', 'measure'],
            'coupling_map': self._generate_heavy_hex_coupling(27)
        }
        
        # IBM Quantum Hummingbird r2 (16 qubit) - 'ibmq_guadalupe'
        guadalupe = {
            'gate_costs': {
                'id': 35.5,
                'x': 35.5,
                'sx': 35.5,
                'rz': 0,  # Virtual gate
                'cx': 300.0,
                'measure': 1000.0,
                'reset': 1000.0
            },
            'error_rates': {
                'id': 0.0001,
                'x': 0.0005,
                'sx': 0.0005,
                'rz': 0.0001,  # Virtual gate
                'cx': 0.012,  # Typical two-qubit gate error for Hummingbird r2
                'measure': 0.025,
                'reset': 0.025
            },
            'basis_gates': ['id', 'rz', 'sx', 'x', 'cx', 'reset', 'measure'],
            'coupling_map': self._generate_heavy_hex_coupling(16)
        }
        
        # IBM Quantum Eagle r1 (127 qubit) - 'ibm_washington'
        washington = {
            'gate_costs': {
                'id': 35.5,
                'x': 35.5,
                'sx': 35.5,
                'rz': 0,  # Virtual gate
                'ecr': 300.0,
                'measure': 1000.0,
                'reset': 1000.0
            },
            'error_rates': {
                'id': 0.0001,
                'x': 0.0003,
                'sx': 0.0003,
                'rz': 0.0001,  # Virtual gate
                'ecr': 0.006,  # Typical two-qubit gate error for Eagle r1
                'measure': 0.015,
                'reset': 0.015
            },
            'basis_gates': ['id', 'rz', 'sx', 'x', 'ecr', 'reset', 'measure'],
            'coupling_map': self._generate_heavy_hex_coupling(127)
        }
        
        # Add backends to the dictionary
        self.backends = {
            'ibm_sherbrooke': sherbrooke,
            'ibm_cairo': cairo,
            'ibmq_guadalupe': guadalupe,
            'ibm_washington': washington
        }
    
    def _generate_heavy_hex_coupling(self, num_qubits):
        """
        Generate a simplified heavy-hex coupling map for the given number of qubits.
        This is an approximation of the IBM Quantum coupling maps.
        
        Args:
            num_qubits (int): Number of qubits.
            
        Returns:
            list: List of qubit pairs representing the coupling map.
        """
        # This is a simplified model of the heavy-hex lattice
        # Real IBM Quantum coupling maps are more complex
        coupling_map = []
        
        # For each qubit, connect to neighbors in a heavy-hex pattern
        for i in range(num_qubits):
            # Connect to right neighbor (if not at edge)
            if (i + 1) % 8 != 0 and (i + 1) < num_qubits:
                coupling_map.append([i, i + 1])
                coupling_map.append([i + 1, i])
            
            # Connect to neighbor below (if exists)
            if i + 8 < num_qubits:
                coupling_map.append([i, i + 8])
                coupling_map.append([i + 8, i])
                
            # Add some diagonal connections for the "heavy" part of heavy-hex
            if i % 16 == 0 and i + 9 < num_qubits:
                coupling_map.append([i, i + 9])
                coupling_map.append([i + 9, i])
            if i % 16 == 7 and i + 7 < num_qubits:
                coupling_map.append([i, i + 7])
                coupling_map.append([i + 7, i])
        
        return coupling_map
    
    def get_backend_data(self, backend_name):
        """
        Get data for a specific backend.
        
        Args:
            backend_name (str): The name of the backend.
            
        Returns:
            dict: Data for the backend, or None if not found.
        """
        return self.backends.get(backend_name, None)
    
    def get_available_backends(self):
        """
        Get a list of available backends.
        
        Returns:
            list: List of available backend names.
        """
        return list(self.backends.keys())
    
    def get_basis_gates(self, backend_name):
        """
        Get the basis gates for a specific backend.
        
        Args:
            backend_name (str): The name of the backend.
            
        Returns:
            list: List of basis gates, or None if backend not found.
        """
        backend_data = self.get_backend_data(backend_name)
        if backend_data:
            return backend_data.get('basis_gates', None)
        return None
    
    def get_coupling_map(self, backend_name):
        """
        Get the coupling map for a specific backend.
        
        Args:
            backend_name (str): The name of the backend.
            
        Returns:
            list: Coupling map, or None if backend not found.
        """
        backend_data = self.get_backend_data(backend_name)
        if backend_data:
            return backend_data.get('coupling_map', None)
        return None
    
    def get_gate_costs(self, backend_name):
        """
        Get the gate costs for a specific backend.
        
        Args:
            backend_name (str): The name of the backend.
            
        Returns:
            dict: Gate costs, or None if backend not found.
        """
        backend_data = self.get_backend_data(backend_name)
        if backend_data:
            return backend_data.get('gate_costs', None)
        return None
    
    def get_error_rates(self, backend_name):
        """
        Get the error rates for a specific backend.
        
        Args:
            backend_name (str): The name of the backend.
            
        Returns:
            dict: Error rates, or None if backend not found.
        """
        backend_data = self.get_backend_data(backend_name)
        if backend_data:
            return backend_data.get('error_rates', None)
        return None
    
    def save_to_file(self, filename):
        """
        Save the IBM quantum data to a JSON file.
        
        Args:
            filename (str): Path to the JSON file.
        """
        try:
            with open(filename, 'w') as f:
                json.dump(self.backends, f, indent=2)
                
        except Exception as e:
            print(f"Error saving data to file {filename}: {str(e)}")
    
    def load_from_file(self, filename):
        """
        Load IBM quantum data from a JSON file.
        
        Args:
            filename (str): Path to the JSON file.
        """
        try:
            with open(filename, 'r') as f:
                self.backends = json.load(f)
                
        except Exception as e:
            print(f"Error loading data from file {filename}: {str(e)}")
            print("Using default data instead.")
            self._init_default_data()
    
    def __str__(self):
        """String representation of the IBM quantum data."""
        return f"IBMQuantumData with {len(self.backends)} backends: {', '.join(self.backends.keys())}"
