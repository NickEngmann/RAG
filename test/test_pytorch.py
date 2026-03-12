import subprocess
import sys

def test_pytorch():
    """Test pytorch functionality."""
    result = subprocess.run(
        [sys.executable, 'test/pytorch-test.py'],
        capture_output=True,
        text=True,
        timeout=30
    )
    assert result.returncode == 0, f"pytorch test failed: {result.stderr}"
    assert "PyTorch test completed successfully." in result.stdout
