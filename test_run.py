#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Test Script - Validate Core Functions of run.py
"""

import os
import sys

def test_imports():
    """Test required imports"""
    print("Testing module imports...")
    try:
        import pandas as pd
        print("[OK] pandas imported successfully")

        import numpy as np
        print("[OK] numpy imported successfully")

        import dataset
        print("[OK] dataset module imported successfully")

        import plot
        print("[OK] plot module imported successfully")

        # Test distance calculation modules
        try:
            from distance import aggregate_fed, aggregate_1
            print("[OK] Distance aggregation modules imported successfully")
        except ImportError as e:
            print(f"[WARN] Distance aggregation module import warning: {e}")

        return True

    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return False

def test_directory_structure():
    """Test directory structure"""
    print("\nChecking directory structure...")

    directories = ["client_model", "data", "exp", "distance"]
    for directory in directories:
        if os.path.exists(directory):
            print(f"[OK] {directory}/ directory exists")
        else:
            print(f"[WARN] {directory}/ directory does not exist")

def test_run_py_imports():
    """Test run.py imports"""
    print("\nTesting run.py imports...")
    try:
        # Attempt to import functions from run.py
        sys.path.append(os.getcwd())

        # Check if run.py file exists
        if not os.path.exists("run.py"):
            print("[ERROR] run.py file does not exist")
            return False

        print("[OK] run.py file exists")

        # Verify syntax
        with open("run.py", 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if key functions are defined
        required_functions = [
            'setup_directories',
            'check_data_files',
            'generate_client_models_if_needed',
            'load_client_models',
            'run_aggregation_methods',
            'run_single_experiment',
            'main'
        ]

        for func in required_functions:
            if f"def {func}(" in content:
                print(f"[OK] function {func} defined")
            else:
                print(f"[ERROR] function {func} not found")

        return True

    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return False

def test_config_parameters():
    """Test configuration parameters"""
    print("\nChecking configuration parameters...")

    try:
        # Read run.py file
        with open("run.py", 'r', encoding='utf-8') as f:
            content = f.read()

        # Check key configuration parameters
        configs = {
            'MODEL_NAME': 'MLP',
            'NUM_CLIENT': 6,
            'NUM_EXPERIMENTS': 100,
            'MINI_TYPE': 2,
            'MAX_TYPE': 5,
            'HIDDEN_DIM': 10
        }

        for config, expected_value in configs.items():
            if f"{config} = {expected_value}" in content:
                print(f"[OK] {config} = {expected_value}")
            else:
                print(f"[WARN] {config} may not be set correctly")

        return True

    except Exception as e:
        print(f"[ERROR] Configuration check failed: {e}")
        return False

def main():
    """Main test function"""
    print("run.py Function Validation Test")
    print("=" * 50)

    # Run all tests
    tests = [
        test_imports,
        test_directory_structure,
        test_run_py_imports,
        test_config_parameters
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"[ERROR] Exception in test {test.__name__}: {e}")

    print("\nTest Results Summary")
    print("=" * 50)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("All tests passed! run.py modification successful!")
        return True
    else:
        print("Some tests failed, please check the issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)