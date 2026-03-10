#!/usr/bin/env python3

import subprocess
import sys
import os

def run_test(script_name):
    """Run a single test script and return the result."""
    try:
        # Run from the test directory
        test_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(test_dir, script_name)
        
        result = subprocess.run(['python', script_path], capture_output=True, text=True, timeout=60, cwd=test_dir)
        print(f"\n{'='*60}")
        print(f"Running {script_name}...")
        print(f"{'='*60}")
        print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"Test {script_name} timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"Error running {script_name}: {e}")
        return False

def main():
    """Run all test scripts."""
    test_scripts = [
        'tqdm-test.py',
        'pytorch-test.py',
        'sentence-test.py',
        'elastic-test.py',
        'gc-test.py'
    ]
    
    results = []
    for script in test_scripts:
        passed = run_test(script)
        results.append((script, passed))
    
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    
    for script, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{script}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\nAll tests passed!")
        return 0
    else:
        print(f"\n{sum(1 for _, passed in results if not passed)} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
