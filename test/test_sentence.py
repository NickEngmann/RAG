import subprocess
import sys

def test_sentence():
    """Test sentence-transformers functionality."""
    result = subprocess.run(
        [sys.executable, 'test/sentence-test.py'],
        capture_output=True,
        text=True,
        timeout=30
    )
    assert result.returncode == 0, f"sentence test failed: {result.stderr}"
    assert "sentence_transformers test completed successfully" in result.stdout
