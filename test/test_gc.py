import pytest
import gc

def test_gc():
    """Test that gc (Garbage Collector) library is installed and working."""
    # Create some objects
    large_list = [i for i in range(100000)]  # Reduced size for faster tests
    del large_list

    # Get count of objects before collection
    count_before = len(gc.get_objects())

    # Perform garbage collection
    collected = gc.collect()

    # Get count of objects after collection
    count_after = len(gc.get_objects())

    print(f"Objects before GC: {count_before}")
    print(f"Objects after GC: {count_after}")
    print(f"Objects collected: {collected}")

    # Verify gc.collect() returns a non-negative integer
    assert isinstance(collected, int)
    assert collected >= 0
    
    print("gc test completed successfully")