#!/usr/bin/env python3
"""Basic tests for RAG system without external dependencies."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that main module can be imported."""
    try:
        import rag_system
        print("✓ rag_system module imported successfully")
        assert True
    except Exception as e:
        print(f"✗ Failed to import rag_system: {e}")
        assert False, f"Failed to import rag_system: {e}"


def test_classes_exist():
    """Test that key classes exist in rag_system."""
    try:
        import rag_system
        
        # Check for main classes
        classes_to_check = ['RAGSystem']
        for cls_name in classes_to_check:
            assert hasattr(rag_system, cls_name), f"{cls_name} class not found"
            print(f"✓ {cls_name} class exists")
    except Exception as e:
        assert False, f"Error checking classes: {e}"


def test_functions_exist():
    """Test that key functions exist in rag_system."""
    try:
        import rag_system
        
        # Check for main functions
        functions_to_check = ['main']
        for func_name in functions_to_check:
            assert hasattr(rag_system, func_name), f"{func_name} function not found"
            print(f"✓ {func_name} function exists")
    except Exception as e:
        assert False, f"Error checking functions: {e}"


def run_all_tests():
    """Run all basic tests."""
    print("Running basic RAG system tests...\n")
    
    test_imports()
    test_classes_exist()
    test_functions_exist()
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    run_all_tests()
