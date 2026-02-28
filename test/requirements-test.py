import subprocess
import os

def run_test(script_name):
    try:
        script_path = os.path.join(os.path.dirname(__file__), script_name)
        result = subprocess.run(['python', script_path], capture_output=True, text=True)
        print(f"Running {script_name}...")
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"Errors in {script_name}:\n{result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {script_name}: {e}")
        return False

if __name__ == "__main__":
    test_scripts = [
        'tqdm-test.py',
        'pytorch-test.py',
        'sentence-test.py',
        'elastic-test.py',
        'gc-test.py'
    ]

    passed = 0
    failed = 0
    errors = 0

    for script in test_scripts:
        result = run_test(script)
        if result:
            passed += 1
            print(f"{script}: PASSED\n")
        else:
            failed += 1
            print(f"{script}: FAILED\n")

    print(f"\nTest Results: {passed} passed, {failed} failed, {errors} errors")
