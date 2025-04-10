# Quantum Circuit Optimizer

This project implements various approaches to quantum circuit optimization using reinforcement learning and rule-based techniques.

## Overview

Quantum circuit optimization is essential for practical quantum computing, as it reduces the number of gates and circuit depth, leading to more efficient execution on real quantum hardware with limited coherence times.

This project provides multiple optimization approaches:

1. **GNN-based Reinforcement Learning Optimizer** - Uses a Graph Neural Network (GNN) to learn optimization strategies through reinforcement learning.
2. **Rule-based Optimizer** - Implements specific optimization rules like redundant gate elimination.
3. **Comprehensive Optimizer** - Combines multiple optimization techniques for better results.

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

### Reward Functions

The project includes several reward functions for reinforcement learning:

- `environment/advanced_reward.py` - Advanced reward function with multiple components
- `environment/improved_reward.py` - Improved reward function with additional components

## Usage

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

## Future Improvements

1. **Enhanced Gate Commutation** - Implement more sophisticated gate commutation rules
2. **Gate Decomposition** - Add support for decomposing complex gates into simpler ones
3. **Template Matching** - Implement template-based optimization techniques
4. **Improved GNN Architecture** - Explore more advanced GNN architectures for better learning
5. **Multi-objective Optimization** - Consider multiple objectives like gate count, depth, and error rates

## Dependencies

- Python 3.8+
- Qiskit
- PyTorch
- NetworkX
- NumPy
- Matplotlib
- Gymnasium (formerly Gym)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd quantum-circuit-optimizer

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
