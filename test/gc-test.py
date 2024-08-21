#!/usr/bin/env python3

import gc
import sys

print("Testing gc (Garbage Collector)...")

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

print("gc test completed successfully")