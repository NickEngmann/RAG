#!/usr/bin/env python3

import subprocess
import os
import sys

def run_test(script_name):
    """Run a single test script and capture output."""
    try:
        result = subprocess.run(['python', script_name], capture_output=True, text=True)
        print(f"Running {script_name}...\n")
        print(result.stdout)
        if result.stderr:
            print(f"Errors in {script_name}:\n{result.stderr}")
    except Exception as e:
        print(f"Failed to run {script_name}: {e}")

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    test_scripts = [
        'tqdm-test.py',
        'pytorch-test.py',
        'sentence-test.py',
        'elastic-test.py',
        'gc-test.py'
    ]
    
    print("=" * 50)
    print("Running all RAG system tests")
    print("=" * 50)
    
    for script in test_scripts:
        script_path = os.path.join(script_dir, script)
        if os.path.exists(script_path):
            run_test(script_path)
        else:
            print(f"Warning: {script_path} not found, skipping...")
    
    print("\n" + "=" * 50)
    print("All tests completed")
    print("=" * 50)
