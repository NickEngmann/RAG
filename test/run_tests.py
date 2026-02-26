#!/usr/bin/env python3
"""Comprehensive test runner for RAG system."""

import subprocess
import sys
import os


def run_test_with_fallback(script_name, test_dir, optional=False):
    """Run a test script with fallback for missing dependencies."""
    script_path = os.path.join(test_dir, script_name)
    
    if not os.path.exists(script_path):
        print(f"⚠ {script_name}: File not found")
        return None
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print(f"\n{'='*60}")
        print(f"Running {script_name}...")
        print(f"{'='*60}")
        
        if result.returncode == 0:
            print(f"✓ {script_name}: PASSED")
            print(result.stdout)
            return True
        else:
            # Check if it's a dependency issue
            output = result.stderr + result.stdout
            if 'ModuleNotFoundError' in output or 'ImportError' in output:
                if optional:
                    print(f"⚠ {script_name}: SKIPPED (optional dependency missing)")
                    print(f"  Details: {output.strip().split(chr(10))[-1]}")
                    return None
                else:
                    print(f"⚠ {script_name}: FAILED (dependency issue)")
                    print(f"  Details: {output.strip().split(chr(10))[-1]}")
                    return False
            else:
                print(f"✗ {script_name}: FAILED")
                print(f"  Error: {output}")
                return False
                
    except subprocess.TimeoutExpired:
        print(f"✗ {script_name}: TIMEOUT")
        return False
    except Exception as e:
        print(f"✗ {script_name}: ERROR - {e}")
        return False


def run_basic_tests():
    """Run basic tests that don't require external dependencies."""
    print("\n" + "="*60)
    print("BASIC TESTS (no external dependencies)")
    print("="*60)
    
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Basic test that checks imports without external deps
    result = subprocess.run(
        [sys.executable, os.path.join(test_dir, 'test_rag_basic.py')],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    return result.returncode == 0


def run_all_tests():
    """Run all tests with appropriate handling."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("RAG System Test Suite")
    print("="*60)
    
    # Track results
    results = {
        'passed': 0,
        'failed': 0,
        'skipped': 0
    }
    
    # Run basic tests first
    basic_passed = run_basic_tests()
    if basic_passed:
        results['passed'] += 1
    else:
        results['failed'] += 1
    
    # Define test scripts with their optional status
    test_scripts = [
        ('gc-test.py', False),
        ('tqdm-test.py', False),
        ('sentence-test.py', True),  # Optional
        ('pytorch-test.py', True),   # Optional
        ('elastic-test.py', True),   # Optional (requires ES)
    ]
    
    # Run each test
    for script_name, optional in test_scripts:
        result = run_test_with_fallback(script_name, test_dir, optional)
        
        if result is True:
            results['passed'] += 1
        elif result is False:
            results['failed'] += 1
        else:  # None = skipped
            results['skipped'] += 1
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed:  {results['passed']}")
    print(f"Failed:  {results['failed']}")
    print(f"Skipped: {results['skipped']}")
    print(f"Total:   {sum(results.values())}")
    print("="*60)
    
    return results['failed'] == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
