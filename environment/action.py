"""
Circuit Action
============

This module provides the action representation for the quantum circuit optimization environment.
"""

import numpy as np
from enum import Enum, auto

class ActionType(Enum):
    """Enumeration of possible action types for circuit optimization."""
    DECOMPOSE = auto()          # Decompose a gate into simpler gates
    OPTIMIZE_1Q = auto()        # Optimize single-qubit gates
    CANCEL_2Q = auto()          # Cancel adjacent two-qubit gates
    CONSOLIDATE = auto()        # Consolidate adjacent gates
    INSERT_GATE = auto()        # Insert a gate
    REMOVE_GATE = auto()        # Remove a gate
    REPLACE_GATE = auto()       # Replace a gate with another gate
    COMMUTE_GATES = auto()      # Commute two gates
    NO_OP = auto()              # No operation (do nothing)

class CircuitAction:
    """
    Represents an action that can be applied to a quantum circuit during optimization.
    """
    
    def __init__(self, action_type, **params):
        """
        Initialize a circuit action.
        
        Args:
            action_type (ActionType): The type of action to perform.
            **params: Additional parameters for the action.
                For DECOMPOSE: basis_gates (list)
                For INSERT_GATE: gate (str), qubits (list), position (int, optional)
                For REMOVE_GATE: position (int)
                For REPLACE_GATE: position (int), new_gate (str), new_qubits (list, optional)
                For COMMUTE_GATES: position1 (int), position2 (int)
        """
        self.action_type = action_type
        self.params = params
    
    @classmethod
    def from_index(cls, index, num_actions, num_qubits, available_gates):
        """
        Create an action from an index (for use with discrete action spaces).
        
        Args:
            index (int): The index of the action.
            num_actions (int): The total number of possible actions.
            num_qubits (int): The number of qubits in the circuit.
            available_gates (list): List of available gates.
            
        Returns:
            CircuitAction: The action corresponding to the index.
        """
        # Calculate the number of actions for each action type
        num_action_types = len(ActionType)
        
        # Basic actions (DECOMPOSE, OPTIMIZE_1Q, CANCEL_2Q, CONSOLIDATE, NO_OP)
        basic_actions = 5
        
        # Gate insertion actions: For each gate and each qubit combination
        num_insert_actions = len(available_gates) * (num_qubits + num_qubits * (num_qubits - 1) // 2)
        
        # Gate removal actions: One for each position (assume max 100 gates)
        num_remove_actions = 100
        
        # Gate replacement actions: For each position and each gate
        num_replace_actions = 100 * len(available_gates)
        
        # Gate commutation actions: For each pair of adjacent positions (assume max 100 gates)
        num_commute_actions = 99
        
        # Determine the action type based on the index
        if index < basic_actions:
            # Basic actions
            if index == 0:
                return cls(ActionType.DECOMPOSE, basis_gates=['u1', 'u2', 'u3', 'cx'])
            elif index == 1:
                return cls(ActionType.OPTIMIZE_1Q)
            elif index == 2:
                return cls(ActionType.CANCEL_2Q)
            elif index == 3:
                return cls(ActionType.CONSOLIDATE)
            else:
                return cls(ActionType.NO_OP)
        elif index < basic_actions + num_insert_actions:
            # INSERT_GATE actions
            idx = index - basic_actions
            gate_idx = idx % len(available_gates)
            qubit_idx = idx // len(available_gates)
            
            gate = available_gates[gate_idx]
            
            if qubit_idx < num_qubits:
                # Single-qubit gate
                qubits = [qubit_idx]
            else:
                # Two-qubit gate
                qubit_idx -= num_qubits
                q1 = qubit_idx // (num_qubits - 1)
                q2 = qubit_idx % (num_qubits - 1)
                if q2 >= q1:
                    q2 += 1
                qubits = [q1, q2]
            
            return cls(ActionType.INSERT_GATE, gate=gate, qubits=qubits)
        elif index < basic_actions + num_insert_actions + num_remove_actions:
            # REMOVE_GATE actions
            position = index - (basic_actions + num_insert_actions)
            return cls(ActionType.REMOVE_GATE, position=position)
        elif index < basic_actions + num_insert_actions + num_remove_actions + num_replace_actions:
            # REPLACE_GATE actions
            idx = index - (basic_actions + num_insert_actions + num_remove_actions)
            gate_idx = idx % len(available_gates)
            position = idx // len(available_gates)
            
            gate = available_gates[gate_idx]
            
            return cls(ActionType.REPLACE_GATE, position=position, new_gate=gate)
        else:
            # COMMUTE_GATES actions
            idx = index - (basic_actions + num_insert_actions + num_remove_actions + num_replace_actions)
            position1 = idx
            position2 = idx + 1
            
            return cls(ActionType.COMMUTE_GATES, position1=position1, position2=position2)
    
    @classmethod
    def get_num_actions(cls, num_qubits, available_gates):
        """
        Get the total number of possible actions.
        
        Args:
            num_qubits (int): The number of qubits in the circuit.
            available_gates (list): List of available gates.
            
        Returns:
            int: The total number of possible actions.
        """
        # Basic actions (DECOMPOSE, OPTIMIZE_1Q, CANCEL_2Q, CONSOLIDATE, NO_OP)
        basic_actions = 5
        
        # Gate insertion actions: For each gate and each qubit combination
        num_insert_actions = len(available_gates) * (num_qubits + num_qubits * (num_qubits - 1) // 2)
        
        # Gate removal actions: One for each position (assume max 100 gates)
        num_remove_actions = 100
        
        # Gate replacement actions: For each position and each gate
        num_replace_actions = 100 * len(available_gates)
        
        # Gate commutation actions: For each pair of adjacent positions (assume max 100 gates)
        num_commute_actions = 99
        
        return basic_actions + num_insert_actions + num_remove_actions + num_replace_actions + num_commute_actions
    
    def apply(self, state):
        """
        Apply the action to a circuit state.
        
        Args:
            state (CircuitState): The circuit state to apply the action to.
            
        Returns:
            CircuitState: A new circuit state with the action applied.
        """
        from quantum_circuit.transformations import CircuitTransformer
        from environment.state import CircuitState
        
        # Get the circuit from the state
        circuit = state.circuit
        
        # Apply the action based on the action type
        if self.action_type == ActionType.DECOMPOSE:
            basis_gates = self.params.get('basis_gates', ['u1', 'u2', 'u3', 'cx'])
            new_circuit = CircuitTransformer.decompose_to_basis_gates(circuit, basis_gates)
        elif self.action_type == ActionType.OPTIMIZE_1Q:
            new_circuit = CircuitTransformer.optimize_1q_gates(circuit)
        elif self.action_type == ActionType.CANCEL_2Q:
            new_circuit = CircuitTransformer.cancel_two_qubit_gates(circuit)
        elif self.action_type == ActionType.CONSOLIDATE:
            new_circuit = CircuitTransformer.consolidate_blocks(circuit)
        elif self.action_type == ActionType.INSERT_GATE:
            gate = self.params.get('gate')
            qubits = self.params.get('qubits')
            position = self.params.get('position')
            new_circuit = CircuitTransformer.insert_gate(circuit, gate, qubits, position)
        elif self.action_type == ActionType.REMOVE_GATE:
            position = self.params.get('position')
            new_circuit = CircuitTransformer.remove_gate(circuit, position)
        elif self.action_type == ActionType.REPLACE_GATE:
            position = self.params.get('position')
            new_gate = self.params.get('new_gate')
            new_qubits = self.params.get('new_qubits')
            new_circuit = CircuitTransformer.replace_gate(circuit, position, new_gate, new_qubits)
        elif self.action_type == ActionType.COMMUTE_GATES:
            position1 = self.params.get('position1')
            position2 = self.params.get('position2')
            new_circuit = CircuitTransformer.commute_gates(circuit, position1, position2)
        elif self.action_type == ActionType.NO_OP:
            # No operation, return the original circuit
            return state.copy()
        else:
            raise ValueError(f"Unknown action type: {self.action_type}")
        
        # Create a new state with the transformed circuit
        new_state = CircuitState(new_circuit, state.cost_database)
        
        return new_state
    
    def __str__(self):
        """String representation of the action."""
        return f"CircuitAction({self.action_type.name}, {self.params})"
    
    def __repr__(self):
        """Representation of the action."""
        return str(self)
