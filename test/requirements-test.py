import subprocess
import os

def run_test(script_name):
    try:
        script_path = os.path.join(os.path.dirname(__file__), script_name)
        result = subprocess.run(['python', script_path], capture_output=True, text=True)
        print(f"Running {script_name}...\n")
        print(result.stdout)
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

    for script in test_scripts:
        run_test(script)