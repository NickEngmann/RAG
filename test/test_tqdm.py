import pytest
from tqdm import tqdm
import time

def test_tqdm():
    """Test that tqdm library is installed and working."""
    results = []
    for i in tqdm(range(10), desc="Processing"):
        results.append(i)
        time.sleep(0.01)  # Reduced sleep time for faster tests
    
    assert len(results) == 10
    assert results == list(range(10))
    print("tqdm test completed successfully")