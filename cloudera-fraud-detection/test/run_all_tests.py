#!/usr/bin/env python3
"""
Run all tests for the fraud detection pipeline
"""
import os
import sys
import subprocess
import time

def run_test(test_name, script_name):
    """Run a single test script and return success status"""
    print(f"\n{'='*60}")
    print(f"Running {test_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the test script
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__)
        )
        
        # Print output
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✓ {test_name} completed successfully in {elapsed_time:.2f}s")
            return True
        else:
            print(f"\n✗ {test_name} failed after {elapsed_time:.2f}s")
            return False
            
    except Exception as e:
        print(f"\n✗ Error running {test_name}: {str(e)}")
        return False

def main():
    """Run all tests in sequence"""
    print("="*60)
    print("Fraud Detection Pipeline Test Suite")
    print("="*60)
    print("\nThis will test the complete fraud detection pipeline:")
    print("1. Feature Engineering")
    print("2. Model Training")
    print("3. Transaction Scoring")
    
    test_dir = os.path.dirname(__file__)
    
    # Define tests in order
    tests = [
        ("Feature Engineering Test", "test_feature_engineering.py"),
        ("Model Training Test", "test_train_model.py"),
        ("Transaction Scoring Test", "test_score_transactions.py")
    ]
    
    # Track results
    results = []
    total_start = time.time()
    
    # Run each test
    for test_name, script_name in tests:
        script_path = os.path.join(test_dir, script_name)
        if not os.path.exists(script_path):
            print(f"\n✗ Test script not found: {script_path}")
            results.append((test_name, False))
            continue
            
        success = run_test(test_name, script_path)
        results.append((test_name, success))
        
        # Stop if a test fails (since later tests depend on earlier ones)
        if not success:
            print("\nStopping test suite due to failure.")
            break
    
    # Print summary
    total_time = time.time() - total_start
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        symbol = "✓" if success else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print(f"Total time: {total_time:.2f}s")
    
    if passed == total:
        print("\n✓ All tests passed! The fraud detection pipeline is working correctly.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())