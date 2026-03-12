import pytest
import os

def test_requirements():
    """Test that all required test files exist."""
    test_files = [
        'test_pytorch.py',
        'test_tqdm.py',
        'test_sentence.py',
        'test_elastic.py',
        'test_gc.py'
    ]
    
    for test_file in test_files:
        test_path = os.path.join(os.path.dirname(__file__), test_file)
        assert os.path.exists(test_path), f"Test file {test_file} not found"
        # Verify it's a valid Python file
        with open(test_path, 'r') as f:
            code = f.read()
            assert 'def test_' in code or 'import pytest' in code, f"{test_file} is not a valid pytest test file"

if __name__ == '__main__':
    test_requirements()
    print("All required test files exist and are valid pytest files")
