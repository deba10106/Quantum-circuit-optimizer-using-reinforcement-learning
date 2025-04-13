# Quantum Circuit Optimizer with Reinforcement Learning

This project implements a reinforcement learning-based quantum circuit optimizer that balances circuit depth, execution cost, and error rates. The optimizer is designed to work with IBM Quantum computers and uses real gate cost and error rate data to optimize quantum circuits.

## Overview

Quantum circuit optimization is essential for practical quantum computing, as it reduces the number of gates and circuit depth, leading to more efficient execution on real quantum hardware with limited coherence times.

This project provides multiple optimization approaches:

1. **GNN-based Reinforcement Learning Optimizer** - Uses a Graph Neural Network (GNN) to learn optimization strategies through reinforcement learning.
2. **Rule-based Optimizer** - Implements specific optimization rules like redundant gate elimination.
3. **Comprehensive Optimizer** - Combines multiple optimization techniques for better results.

## Features

- Quantum circuit optimization using reinforcement learning
- Consideration of circuit depth, gate costs, and error rates
- Integration with IBM Quantum hardware specifications
- Visualization tools for optimized circuits
- Benchmarking against standard optimization techniques
- **Graph Neural Network (GNN) based state representation**
- **Hierarchical action space decomposition**
- **Advanced reward mechanisms with trend analysis**
- **Circuit representation as directed acyclic graphs (DAGs)**
- **Rule-based optimization techniques for redundant gate elimination**

## Installation 

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

## Dependencies

- Python 3.8+
- Qiskit
- PyTorch
- NetworkX
- NumPy
- Matplotlib
- Gymnasium (formerly Gym)

## Project Structure

- `environment/`: RL environment implementation
  - `quantum_environment.py`: Base RL environment for circuit optimization
  - `action.py`: Action representation for circuit transformations
  - `state.py`: State representation for quantum circuits
  - **`hierarchical_action.py`: Hierarchical action space decomposition**
  - **`gnn_quantum_environment.py`: GNN-based quantum environment**
  - **`advanced_reward.py`: Advanced reward mechanisms**
  - **`improved_reward.py`: Improved reward function with additional components**
- `agent/`: RL agent implementation
  - `dqn_agent.py`: Deep Q-Network agent
  - **`gnn_model.py`: GNN-based models for circuit representation**
  - **`gnn_dqn_agent.py`: GNN-based DQN agent**
  - **`hierarchical_gnn_dqn_agent.py`: Hierarchical GNN-based DQN agent**
- `quantum_circuit/`: Circuit representation and operations
  - **`dag.py`: Directed acyclic graph representation of circuits**
- `cost_database/`: Gate costs and error rates
- `evaluation/`: Evaluation and benchmarking tools
  - `metrics.py`: Metrics for evaluating circuit optimizations
  - `visualization.py`: Visualization tools for circuits and results
  - `benchmarks.py`: Benchmarking tools for comparing optimizers
  - **`gnn_analysis.py`: Analysis tools for GNN-based optimization**
- `examples/`: Example scripts demonstrating usage
  - **`gnn_optimizer_example.py`: Example of GNN-based optimization**
- `utils/`: Utility functions

## Components

### GNN-based Optimizer

The GNN-based optimizer uses reinforcement learning to discover optimization strategies:

- `train_gnn_optimizer.py` - Main training script for the GNN-based optimizer
- `test_gnn_optimizer.py` - Basic test script for the GNN-based optimizer
- `enhanced_test_gnn_optimizer.py` - Enhanced test script with multiple circuit types
- `agent/hierarchical_gnn_dqn_agent.py` - Hierarchical GNN-DQN agent implementation
- `environment/gnn_quantum_environment.py` - Environment for quantum circuit optimization

### Rule-based Optimizers

The rule-based optimizers implement specific optimization rules:

- `robust_circuit_optimizer.py` - Simple optimizer focusing on redundant gate elimination
- `comprehensive_optimizer.py` - Comprehensive optimizer with multiple optimization techniques
- `final_improved_optimizer.py` - Final improved version with additional optimization patterns

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

### GNN-based Optimization

For the advanced GNN-based optimization approach:

```python
from quantum_circuit_optimizer.environment.gnn_quantum_environment import GNNQuantumEnvironment
from quantum_circuit_optimizer.agent.hierarchical_gnn_dqn_agent import HierarchicalGNNDQNAgent
from quantum_circuit_optimizer.cost_database.cost_database import CostDatabase

# Create a cost database
cost_database = CostDatabase()

# Create a GNN-based environment
env = GNNQuantumEnvironment(
    initial_circuit=circuit,
    cost_database=cost_database,
    use_hierarchical_actions=True
)

# Create a hierarchical GNN-based agent
agent = HierarchicalGNNDQNAgent(
    node_feature_dim=env.node_feature_dim,
    num_categories=env.num_categories,
    num_actions_per_category=env.num_actions_per_category,
    gnn_type='gcn'
)

# Train the agent
train_results = agent.train(env, num_steps=1000)

# Optimize a circuit
optimized_circuit = optimize_circuit_with_agent(circuit, agent, env)
```

### Training the GNN-based Optimizer

```bash
python train_gnn_optimizer.py --num_steps 5000 --circuit_type random
```

### Testing the GNN-based Optimizer

```bash
python test_gnn_optimizer.py
```

### Running the Enhanced Test Script

```bash
python enhanced_test_gnn_optimizer.py
```

### Using the Rule-based Optimizers

```bash
python robust_circuit_optimizer.py
python comprehensive_optimizer.py
python final_improved_optimizer.py
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

To run the GNN-based optimization example:

```bash
python examples/gnn_optimizer_example.py
```

Or use the training script with various options:

```bash
python train_gnn_optimizer.py --mode train --agent hierarchical --gnn_type gcn --num_steps 5000
```

## Optimization Results

The optimizers have been tested on various circuit types with known optimization opportunities:

1. **Redundant Gate Circuits** - Circuits with redundant gates that cancel each other out (e.g., H-H pairs)
2. **Commutable Gate Circuits** - Circuits with gates that can be reordered to reduce depth
3. **Decomposable Gate Circuits** - Circuits with gates that can be decomposed into simpler ones
4. **QFT Circuits** - Quantum Fourier Transform circuits with known optimization opportunities
5. **Mixed Circuits** - Circuits with multiple optimization opportunities

### Performance Summary

| Circuit Type | Depth Reduction | Gate Count Reduction |
|--------------|-----------------|----------------------|
| Redundant Gates | 3.00% | 9.00% |
| Commutable Gates | 0.00% | 0.00% |
| Decomposable Gates | 0.00% | 0.00% |
| QFT | 1.00% | 1.00% |
| Mixed Circuit | 3.00% | 4.00% |
| Complex Circuit | 2.00% | 4.00% |

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

### Advanced Reward Mechanisms

The enhanced optimizer introduces advanced reward mechanisms that go beyond immediate rewards:

#### Trend-Based Rewards

The trend-based reward component analyzes the trajectory of optimization over time:

$$R_{trend}(s) = w_d \cdot T_d(s) + w_c \cdot T_c(s) + w_e \cdot T_e(s)$$

where $T_d$, $T_c$, and $T_e$ are the trends of depth, cost, and error over recent states.

#### Potential Future Rewards

The potential reward component considers possible future improvements:

$$R_{potential}(s, S_{next}) = \lambda \cdot \max_{s' \in S_{next}} R(s', s)$$

where $\lambda$ is the lookahead factor and $S_{next}$ is the set of potential next states.

#### Combined Advanced Reward

The total advanced reward combines immediate, trend, and potential components:

$$R_{advanced}(s, s_0, S_{next}) = R(s, s_0) + R_{trend}(s) + R_{potential}(s, S_{next})$$

This allows the agent to accept temporary performance decreases if they lead to better long-term optimization.

### Implementation

The reward mechanism is implemented in the `QuantumEnvironment` class, where the `_calculate_reward` method computes the reward based on the current and original circuit states. The agent learns to optimize circuits by maximizing this reward function through a Deep Q-Network (DQN) architecture.

## Graph Neural Network (GNN) Approach

### Circuit Representation as Graphs

Quantum circuits are represented as directed acyclic graphs (DAGs) where:
- Nodes represent quantum gates or operations
- Edges represent qubit connections or operation dependencies
- Node features encode gate type, parameters, and position information
- Edge features represent the type of connection between operations

### GNN Architecture

The GNN-based approach uses:
- Graph Convolutional Networks (GCN) or Graph Attention Networks (GAT)
- Node-level feature extraction for capturing gate properties
- Message passing between connected gates to understand circuit structure
- Global pooling to obtain a circuit-level representation

### Hierarchical Action Space

The hierarchical action space decomposes circuit transformations into:

1. **Action Categories**:
   - Local transformations (affecting single or adjacent gates)
   - Global transformations (affecting circuit structure)
   - Commutation-based transformations
   - Identity-based transformations

2. **Specific Actions**:
   - Within each category, specific transformations are selected
   - This reduces the action space complexity from O(n²) to O(n)

### Advantages of the GNN Approach

- **Structural Understanding**: Captures the topological structure of quantum circuits
- **Invariance Properties**: Provides invariance to certain circuit transformations
- **Scalability**: Handles circuits of varying sizes and complexities
- **Transfer Learning**: Knowledge can transfer between different circuit types

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

| Feature | RL-based Optimizer | Qiskit's Optimizer | GNN-based Optimizer |
|---------|-------------------|-------------------|-------------------|
| **Methodology** | Reinforcement learning | Deterministic, rule-based | GNN + Hierarchical RL |
| **Exploration** | Dynamic, adaptive | Fixed transformation sequence | Structure-aware exploration |
| **Optimization Target** | Multi-objective (depth, cost, error) | Single objective per level | Multi-objective with trend analysis |
| **Adaptability** | Can discover non-obvious patterns | Limited to predefined rules | Can discover complex structural patterns |
| **State Representation** | Feature vectors | N/A | Graph representation |
| **Action Space** | Flat action space | Fixed rules | Hierarchical action space |
| **Scalability** | Limited by state space | Good | Better due to structural encoding |

## Future Improvements

1. **Enhanced Gate Commutation** - Implement more sophisticated gate commutation rules
2. **Gate Decomposition** - Add support for decomposing complex gates into simpler ones
3. **Template Matching** - Implement template-based optimization techniques
4. **Improved GNN Architecture** - Explore more advanced GNN architectures for better learning
5. **Multi-objective Optimization** - Consider multiple objectives like gate count, depth, and error rates

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## References

1. Quarl: A Reinforcement Learning Approach for Quantum Circuit Optimization
2. Graph Neural Networks for Quantum Circuit Analysis
3. Hierarchical Reinforcement Learning for Complex Tasks

## License

This project is licensed under the MIT License - see the LICENSE file for details.
