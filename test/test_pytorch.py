import pytest

def test_pytorch():
    """Test that PyTorch library is installed and working."""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        # Create a simple tensor
        x = torch.rand(5, 3)
        
        # Perform a simple operation
        y = torch.matmul(x, x.t())
        
        assert y.shape == (5, 5)
        print(f"Matrix multiplication result shape: {y.shape}")
        
        print("PyTorch test completed successfully.")
    except ImportError as e:
        pytest.skip(f"PyTorch not installed: {str(e)}")
    except Exception as e:
        pytest.fail(f"PyTorch test failed: {str(e)}")