#!/usr/bin/env python3
"""
gc-test.py - Garbage Collection Testing Script

This script tests Python's garbage collector (gc) functionality, which is critical
for the RAG system when processing large volumes of log data. The gc module helps
manage memory by:
- Collecting unreachable objects
- Cleaning up temporary data structures
- Preventing memory leaks during batch processing

Usage:
    python test/gc-test.py

Dependencies:
    - gc (built-in Python module)

Expected Output:
    - Object counts before and after garbage collection
    - Number of objects collected

Integration with RAG System:
    The RAG system uses garbage collection in:
    - vectorize_logs(): After processing large batches of logs
    - process_logs(): When temporary data structures are no longer needed
    - Memory-intensive operations to prevent resource exhaustion
"""

import gc
import sys

def test_garbage_collection():
    """
    Test garbage collection functionality.
    
    This simulates the memory management patterns used in the RAG system
    when processing large log files and generating embeddings.
    
    The test:
    1. Creates a large list of objects (simulating log batch processing)
    2. Deletes the list
    3. Measures objects before garbage collection
    4. Triggers manual garbage collection
    5. Measures objects after garbage collection
    6. Reports the number of collected objects
    
    Returns:
        bool: True if test completed successfully
    """
    print("Testing gc (Garbage Collector)...")
    
    # Create some objects to simulate log data processing
    # This mimics the large data structures created in vectorize_logs()
    large_list = [i for i in range(1000000)]
    
    # Delete the list to make it eligible for garbage collection
    del large_list
    
    # Get count of objects before collection
    # This represents the state after cleanup of deleted objects
    count_before = len(gc.get_objects())
    
    # Perform garbage collection
    # gc.collect() returns the number of unreachable objects collected
    collected = gc.collect()
    
    # Get count of objects after collection
    count_after = len(gc.get_objects())
    
    print(f"Objects before GC: {count_before}")
    print(f"Objects after GC: {count_after}")
    print(f"Objects collected: {collected}")
    
    # Verify that garbage collection worked
    if collected > 0:
        print("Garbage collection successfully cleaned up objects")
    else:
        print("No unreachable objects were collected")
    
    print("gc test completed successfully")
    return True

if __name__ == "__main__":
    success = test_garbage_collection()
    sys.exit(0 if success else 1)