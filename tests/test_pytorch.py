import pytest


def test_torch():
    """Test PyTorch installation and basic operations."""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        # Create a simple tensor
        x = torch.rand(5, 3)
        print(f"Random tensor:\n{x}")
        
        # Perform a simple operation
        y = torch.matmul(x, x.t())
        print(f"Matrix multiplication result shape: {y.shape}")
        
        assert y.shape == (5, 5), f"Expected shape (5, 5), got {y.shape}"
        
        print("PyTorch test completed successfully.")
    except Exception as e:
        pytest.fail(f"PyTorch test failed. Error: {str(e)}")
