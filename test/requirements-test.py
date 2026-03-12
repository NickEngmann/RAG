import subprocess
import os

def run_test(script_name):
    try:
        # Get the directory where this file is located
        test_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(test_dir, script_name)
        
        result = subprocess.run(['python', script_path], capture_output=True, text=True)
        print(f"Running {script_name}...\n")
        print(result.stdout)
        if result.stderr:
            print(f"Errors in {script_name}:\n{result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {script_name}: {str(e)}")
        return False

if __name__ == "__main__":
    test_scripts = [
        'tqdm-test.py',
        'pytorch-test.py',
        'sentence-test.py',
        'elastic-test.py',
        'gc-test.py'
    ]
    
    results = {}
    for script in test_scripts:
        passed = run_test(script)
        results[script] = passed
        print("-" * 50)
    
    print("\nTest Summary:")
    print(f"Passed: {sum(results.values())}/{len(results)}")
    for script, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {script}")
    
    if all(results.values()):
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed.")
        exit(1)