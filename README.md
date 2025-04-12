# Quantum Circuit Optimizer with Reinforcement Learning

This project implements a reinforcement learning-based quantum circuit optimizer that balances circuit depth, execution cost, and error rates. The optimizer is designed to work with IBM Quantum computers and uses real gate cost and error rate data to optimize quantum circuits.

## Features

- Quantum circuit optimization using reinforcement learning
- Consideration of circuit depth, gate costs, and error rates
- Integration with IBM Quantum hardware specifications
- Visualization tools for optimized circuits
- Benchmarking against standard optimization techniques

## Installation

Clone the latest version (v0.0.1)

1. Clone this repository:
```bash
git clone https://github.com/yourusername/quantum_circuit_optimizer.git
cd quantum_circuit_optimizer
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Project Structure

- `environment/`: RL environment implementation
- `agent/`: RL agent implementation
- `quantum_circuit/`: Circuit representation and operations
- `cost_database/`: Gate costs and error rates
- `evaluation/`: Evaluation and benchmarking tools
- `examples/`: Example scripts demonstrating usage
- `utils/`: Utility functions

## Usage

### Simple Optimization Approach

For a quick demonstration of quantum circuit optimization without the full RL framework, use the `simple_optimize.py` script:

```bash
python simple_optimize.py
```

This script:
- Creates a QFT circuit and a random circuit
- Optimizes them using Qiskit's transpiler
- Calculates metrics for both original and optimized circuits
- Visualizes the optimization results

### Full RL-based Optimization

For the complete RL-based optimization approach:

```python
from quantum_circuit_optimizer import CircuitOptimizer

# Create an optimizer
optimizer = CircuitOptimizer()

# Load a quantum circuit
circuit = load_circuit("my_circuit.qasm")

# Optimize the circuit
optimized_circuit = optimizer.optimize(circuit)

# Evaluate the optimized circuit
results = optimizer.evaluate(optimized_circuit)
```

### Running Examples

To run the example scripts, use:

```bash
python run_example.py examples/optimize_circuit.py
```

Or use the direct optimization script:

```bash
python direct_optimize.py
```

## Reward Mechanism Theory

### Abstract

The quantum circuit optimizer employs a multi-objective reinforcement learning approach to optimize quantum circuits. This section details the theoretical framework of the reward mechanism that guides the learning process, balancing circuit depth, execution cost, error rates, and functional equivalence.

### Introduction

Quantum circuit optimization presents unique challenges due to the need to balance multiple competing objectives: reducing circuit depth, minimizing execution cost, and maintaining low error rates, all while preserving the circuit's functional behavior. Our approach formulates this as a reinforcement learning problem with a carefully designed reward function that quantifies the quality of optimized circuits.

### Methodology

#### Multi-Objective Reward Function

The reward function $R(s, s_0)$ for a state $s$ relative to the original state $s_0$ is defined as:

$$R(s, s_0) = w_d \cdot R_d(s, s_0) + w_c \cdot R_c(s, s_0) + w_e \cdot R_e(s, s_0) + B_{eq}(s, s_0)$$

where:
- $w_d$, $w_c$, and $w_e$ are the weights for depth, cost, and error components (default: 0.3, 0.3, 0.4)
- $R_d$, $R_c$, and $R_e$ are the reward components for depth, cost, and error
- $B_{eq}$ is the equivalence bonus

#### Depth Optimization Component

The depth component rewards reductions in circuit depth (number of sequential operations):

$$R_d(s, s_0) = w_d \cdot \left(\frac{d_0}{max(1, d_s)} - 1.0\right)$$

where $d_0$ and $d_s$ are the depths of the original and current circuits.

#### Cost Optimization Component

The cost component rewards reductions in execution cost based on hardware-specific gate costs:

$$R_c(s, s_0) = w_c \cdot \left(\frac{c_0}{max(1, c_s)} - 1.0\right)$$

where $c_0$ and $c_s$ are the costs of the original and current circuits.

#### Error Reduction Component

The error component rewards improvements in circuit fidelity:

$$R_e(s, s_0) = w_e \cdot \left(\frac{1.0 - e_s}{max(0.01, 1.0 - e_0)} - 1.0\right)$$

where $e_0$ and $e_s$ are the error rates of the original and current circuits.

#### Equivalence Preservation

To ensure that optimizations preserve the circuit's computational purpose, an equivalence bonus $B_{eq}$ is awarded when the optimized circuit remains functionally equivalent to the original:

$$B_{eq}(s, s_0) = 
\begin{cases} 
b_{eq} & \text{if } s \equiv s_0 \\
0 & \text{otherwise}
\end{cases}$$

where $b_{eq}$ is the equivalence bonus value (default: 1.0) and $s \equiv s_0$ denotes functional equivalence.

### Implementation

The reward mechanism is implemented in the `QuantumEnvironment` class, where the `_calculate_reward` method computes the reward based on the current and original circuit states. The agent learns to optimize circuits by maximizing this reward function through a Deep Q-Network (DQN) architecture.

### Theoretical Guarantees

While the reward function does not provide theoretical guarantees of finding the global optimum, empirical results demonstrate its effectiveness in guiding the optimization process toward circuits with improved depth, cost, and error characteristics while maintaining functional equivalence.

### Limitations and Future Work

The current reward mechanism assumes that the objectives (depth, cost, error) are independent, which may not always be the case in real quantum hardware. Future work will explore more sophisticated reward functions that capture the interdependencies between these objectives and incorporate additional metrics such as circuit width and T-count.

## Optimization Principles

The quantum circuit optimizer employs several key principles to reduce circuit depth while preserving computational purpose:

### Circuit Transformation Techniques

1. **Basis Gate Decomposition**: Breaks down complex gates into simpler ones that might reveal optimization opportunities
   
2. **Single-Qubit Gate Optimization**: Combines sequences of single-qubit gates into more efficient representations
   
3. **Commutative Gate Cancellation**: Identifies and removes gates that cancel each other out
   
4. **Gate Insertion and Removal**: Strategically adds or removes gates to enable further optimizations
   
5. **Gate Replacement**: Substitutes gates with equivalent but more efficient alternatives
   
6. **Gate Commutation**: Reorders gates to create optimization opportunities

### Preserving Circuit Equivalence

The optimizer ensures that transformations maintain the circuit's computational purpose through:

1. **Equivalence Checking**: Verifies that optimized circuits produce the same measurement outcomes as the original
   
2. **Functional Transformations**: Uses mathematically equivalent transformations based on quantum circuit identities
   
3. **Reward-Guided Exploration**: Penalizes transformations that break circuit equivalence

## Comparison with Qiskit's Optimizer

### Approach Differences

| Feature | RL-based Optimizer | Qiskit's Optimizer |
|---------|-------------------|-------------------|
| **Methodology** | Reinforcement learning | Deterministic, rule-based |
| **Exploration** | Dynamic, adaptive | Fixed transformation sequence |
| **Optimization Target** | Multi-objective (depth, cost, error) | Single objective per level |
| **Adaptability** | Can discover non-obvious patterns | Limited to predefined rules |

### Advantages

- **Multi-objective Optimization**: Balances circuit depth, execution cost, and error rates with configurable weights
- **Adaptability**: Can potentially discover optimization patterns that rule-based systems might miss
- **Customizability**: Easily modified reward function to prioritize different aspects of circuit quality

### Limitations

- **Computational Overhead**: Requires significant training time compared to deterministic approaches
- **Implementation Dependency**: Relies on Qiskit's basic transformation passes under the hood
- **Maturity**: Experimental approach compared to Qiskit's well-tested optimizer

### When to Use This Optimizer

- **Research**: When exploring new optimization techniques or circuit transformation patterns
- **Hardware-specific Optimization**: When balancing multiple objectives for a specific quantum processor
- **Complex Circuits**: When traditional optimizers fail to find satisfactory solutions

## Troubleshooting

If you encounter import errors:

1. Make sure you've installed the package in development mode:
```bash
pip install -e .
```

2. Try using the simplified optimization approach with `simple_optimize.py`

3. Check that all required dependencies are installed:
```bash
pip install qiskit numpy matplotlib networkx pandas torch
```

## License

MIT
