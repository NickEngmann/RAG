import subprocess
import os
import sys

def run_test(script_name):
    try:
        result = subprocess.run(['python', script_name], capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        print(f"Running {script_name}...")
        print("-" * 50)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        print("-" * 50)
        if result.returncode == 0:
            print(f"✓ {script_name} passed")
        else:
            print(f"✗ {script_name} failed with return code {result.returncode}")
        print()int(result.stdout)
        if result.stderr:
            print(f"Errors in {script_name}:\n{result.stderr}")
    except Exception as e:
        print(f"Failed to run {script_name}: {e}")

if __name__ == "__main__":
    test_scripts = [
        'tqdm-test.py',
        'pytorch-test.py',
        'sentence-test.py',
        'elastic-test.py',
        'gc-test.py'
    ]
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    passed = 0
    failed = 0
    
    print("=" * 50)
    print("Running RAG System Dependency Tests")
    print("=" * 50)
    print()
    
    for script in test_scripts:
        script_path = os.path.join(test_dir, script)
        result = run_test(script_path)
        if result == 0:
            passed += 1
        else:
            failed += 1
    
    print("=" * 50)
    print("Test Summary")
    print("=" * 50)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {len(test_scripts)}")
    
    if failed == 0:
        print("\nAll tests passed! ✓")
    else:
        print(f"\n{failed} test(s) failed. ✗")
        print("\nNote: Some tests may fail if dependencies are not installed.")
        print("Please install required dependencies using: pip install -r requirements.txt")