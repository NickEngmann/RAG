#!/usr/bin/env python3
"""Test runner for RAG system tests."""

import subprocess
import sys
from pathlib import Path

TEST_FILES = [
    "tqdm-test.py",
    "pytorch-test.py",
    "sentence-test.py",
    "elastic-test.py",
    "gc-test.py",
]

def run_tests():
    """Run all test files."""
    test_dir = Path(__file__).parent
    failed_tests = []
    
    for test_file in TEST_FILES:
        test_path = test_dir / test_file
        if test_path.exists():
            print(f"\n{'='*50}")
            print(f"Running {test_file}")
            print(f"{'='*50}")
            try:
                result = subprocess.run(
                    [sys.executable, str(test_path)],
                    cwd=test_dir,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                print(result.stdout)
                if result.stderr:
                    print(result.stderr)
                if result.returncode != 0:
                    failed_tests.append(test_file)
            except subprocess.TimeoutExpired:
                print(f"Test {test_file} timed out")
                failed_tests.append(test_file)
            except Exception as e:
                print(f"Error running {test_file}: {e}")
                failed_tests.append(test_file)
        else:
            print(f"Test file not found: {test_file}")
            failed_tests.append(test_file)
    
    print(f"\n{'='*50}")
    print("Test Summary")
    print(f"{'='*50}")
    print(f"Total tests: {len(TEST_FILES)}")
    print(f"Passed: {len(TEST_FILES) - len(failed_tests)}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print(f"\nFailed tests: {', '.join(failed_tests)}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(run_tests())
