import subprocess
import sys

def test_elastic():
    """Test Elasticsearch functionality."""
    result = subprocess.run(
        [sys.executable, 'test/elastic-test.py'],
        capture_output=True,
        text=True,
        timeout=30
    )
    # Elasticsearch test may fail if no ES running, but should complete
    assert result.returncode == 0, f"elastic test failed: {result.stderr}"
    assert "elasticsearch test completed successfully" in result.stdout
