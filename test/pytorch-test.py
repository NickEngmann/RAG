#!/usr/bin/env python3

import sys

def test_torch():
    print("Testing PyTorch...")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        # Create a simple tensor
        x = torch.rand(5, 3)
        print(f"Random tensor:\n{x}")
        
        # Perform a simple operation
        y = torch.matmul(x, x.t())
        print(f"Matrix multiplication result shape: {y.shape}")
        
        print("PyTorch test completed successfully.")
    except Exception as e:
        print(f"PyTorch test failed. Error: {str(e)}")

if __name__ == "__main__":
    test_torch()