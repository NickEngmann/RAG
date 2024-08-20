#!/usr/bin/env python3

from tqdm import tqdm
import time

print("Testing tqdm...")

for i in tqdm(range(10), desc="Processing"):
    time.sleep(0.1)

print("tqdm test completed successfully")