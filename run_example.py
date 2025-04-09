#!/usr/bin/env python
"""
Runner script for the quantum circuit optimizer examples.
This script ensures that the imports work correctly.
"""

import os
import sys
import importlib.util

def run_example(example_name):
    """Run an example script from the examples directory."""
    # Get the absolute path to the example script
    example_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'examples',
        example_name + '.py'
    )
    
    # Check if the example exists
    if not os.path.exists(example_path):
        print(f"Error: Example '{example_name}' not found at {example_path}")
        return
    
    # Load the example as a module
    spec = importlib.util.spec_from_file_location(example_name, example_path)
    example_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example_module)
    
    # Run the main function if it exists
    if hasattr(example_module, 'main'):
        example_module.main()
    else:
        print(f"Warning: Example '{example_name}' does not have a main() function")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run_example.py <example_name>")
        print("Available examples:")
        examples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples')
        for file in os.listdir(examples_dir):
            if file.endswith('.py'):
                print(f"  - {os.path.splitext(file)[0]}")
        sys.exit(1)
    
    example_name = sys.argv[1]
    run_example(example_name)
