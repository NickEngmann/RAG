import subprocess

def run_test(script_name):
    try:
        result = subprocess.run(['python', script_name], capture_output=True, text=True)
        print(f"Running {script_name}...\n")
        print(result.stdout)
        if result.stderr:
            print(f"Errors in {script_name}:\n{result.stderr}")
    except Exception as e:
        print(f"Failed to run {script_name}: {e}")

if __name__ == "__main__":
    test_scripts = [
        'test/tqdm-test.py',
        'test/pytorch-test.py',
        'test/sentence-test.py',
        'test/elastic-test.py',
        'test/gc-test.py'
    ]

    for script in test_scripts:
        run_test(script)