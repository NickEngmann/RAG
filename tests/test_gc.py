import pytest
import gc


def test_gc():
    """Test garbage collector functionality."""
    # Create some objects
    large_list = [i for i in range(1000000)]
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

    assert isinstance(collected, int), "collected should be an integer"
    assert collected >= 0, "collected should be non-negative"

    print("gc test completed successfully")
