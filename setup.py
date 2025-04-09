from setuptools import setup, find_packages

setup(
    name="quantum_circuit_optimizer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "qiskit>=0.45.0",
        "qiskit-aer>=0.12.0",
        "numpy>=1.22.0",
        "matplotlib>=3.5.0",
        "gymnasium>=0.28.1",
        "torch>=2.0.0",
        "networkx>=3.0",
        "pandas>=1.5.0",
        "tqdm>=4.65.0",
        "plotly>=5.13.0",
        "stable-baselines3>=2.0.0",
    ],
    author="Debasis",
    author_email="example@example.com",
    description="A reinforcement learning-based quantum circuit optimizer",
    keywords="quantum, circuit, optimization, reinforcement learning",
    python_requires=">=3.8",
)
