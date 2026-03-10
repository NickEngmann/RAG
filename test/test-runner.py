#!/usr/bin/env python3
"""Test runner script for RAG system components."""
import subprocess
import os

def run_test(script_name):
    """Run a test script and capture output."""
    try:
        result = subprocess.run(['python', script_name], capture_output=True, text=True, timeout=60)
        print(f"Running {script_name}...\n")
        print(result.stdout)
        if result.stderr:
            print(f"Errors in {script_name}:\n{result.stderr}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"{script_name} timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"Failed to run {script_name}: {e}")
        return False

if __name__ == "__main__":
    test_scripts = [
        'tqdm-test.py',
        'pytorch-test.py',
        'sentence-test.py',
        'elastic-test.py',
        'gc-test.py'
    ]

    print("=" * 60)
    print("RAG System Component Tests")
    print("=" * 60)
    print()

    all_passed = True
    for script in test_scripts:
        script_path = f'test/{script}'
        if not run_test(script_path):
            all_passed = False
        print()

    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed. Check output above.")
        exit(1)
