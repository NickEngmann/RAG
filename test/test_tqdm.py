import subprocess
import sys

def test_tqdm():
    """Test tqdm functionality."""
    result = subprocess.run(
        [sys.executable, 'test/tqdm-test.py'],
        capture_output=True,
        text=True,
        timeout=30
    )
    assert result.returncode == 0, f"tqdm test failed: {result.stderr}"
    assert "tqdm test completed successfully" in result.stdout
