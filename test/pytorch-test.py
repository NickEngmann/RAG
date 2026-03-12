#!/usr/bin/env python3
"""
pytorch-test.py - PyTorch Library Testing Script

This script tests PyTorch installation and basic tensor operations.
PyTorch is a foundational dependency for the RAG system, used in:
- SentenceTransformer model inference (which uses PyTorch internally)
- Custom neural network operations if needed
- GPU acceleration for large-scale processing

Usage:
    python test/pytorch-test.py

Dependencies:
    - torch (PyTorch)

Expected Output:
    - PyTorch version number
    - Tensor creation and manipulation results
    - Device availability (CPU/GPU)
"""

import sys
import torch

def test_torch():
    """
    Test PyTorch installation and basic functionality.
    
    This test verifies:
    1. PyTorch is installed correctly
    2. Tensor operations work as expected
    3. Device availability (CPU/GPU) is detected
    4. Basic tensor manipulation functions
    
    This mirrors the tensor operations used in vectorize_logs() 
    in rag_system.py where embeddings are combined with timestamps.
    """
    print("Testing PyTorch...")
    
    try:
        # Check PyTorch version
        print(f"PyTorch version: {torch.__version__}")
        
        # Test tensor creation
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        print(f"Created tensor: {tensor}")
        
        # Test basic operations
        print(f"Tensor sum: {tensor.sum()}")
        print(f"Tensor mean: {tensor.mean()}")
        
        # Test device availability (CPU/GPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device available: {device}")
        
        # Test tensor operations similar to those in rag_system.py
        # The system uses np.hstack() which is compatible with torch operations
        tensor1 = torch.randn(3, 4)
        tensor2 = torch.randn(3, 1)
        combined = torch.cat((tensor1, tensor2), dim=1)
        print(f"Combined tensor shape: {combined.shape}")
        
        print("PyTorch test completed successfully.")
        return True
        
    except Exception as e:
        print(f"PyTorch test failed. Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_torch()
    sys.exit(0 if success else 1)