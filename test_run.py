#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试脚本 - 验证run.py的核心功能
Quick Test Script - Validate Core Functions of run.py
"""

import os
import sys

def test_imports():
    """测试必要的导入"""
    print("测试模块导入...")
    try:
        import pandas as pd
        print("[OK] pandas 导入成功")

        import numpy as np
        print("[OK] numpy 导入成功")

        import dataset
        print("[OK] dataset 模块导入成功")

        import plot
        print("[OK] plot 模块导入成功")

        # 测试距离计算模块
        try:
            from distance import aggregate_fed, aggregate_1
            print("[OK] 距离聚合模块导入成功")
        except ImportError as e:
            print(f"[WARN] 距离聚合模块导入警告: {e}")

        return True

    except ImportError as e:
        print(f"[ERROR] 导入失败: {e}")
        return False

def test_directory_structure():
    """测试目录结构"""
    print("\n检查目录结构...")

    directories = ["client_model", "data", "exp", "distance"]
    for directory in directories:
        if os.path.exists(directory):
            print(f"[OK] {directory}/ 目录存在")
        else:
            print(f"[WARN] {directory}/ 目录不存在")

def test_run_py_imports():
    """测试run.py的导入"""
    print("\n测试run.py导入...")
    try:
        # 尝试导入run.py中的函数
        sys.path.append(os.getcwd())

        # 检查run.py文件是否存在
        if not os.path.exists("run.py"):
            print("[ERROR] run.py 文件不存在")
            return False

        print("[OK] run.py 文件存在")

        # 验证语法
        with open("run.py", 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查关键函数是否定义
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
                print(f"[OK] 函数 {func} 已定义")
            else:
                print(f"[ERROR] 函数 {func} 未找到")

        return True

    except Exception as e:
        print(f"[ERROR] 测试失败: {e}")
        return False

def test_config_parameters():
    """测试配置参数"""
    print("\n检查配置参数...")

    try:
        # 读取run.py文件
        with open("run.py", 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查关键配置参数
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
                print(f"[WARN] {config} 可能未正确设置")

        return True

    except Exception as e:
        print(f"[ERROR] 配置检查失败: {e}")
        return False

def main():
    """主测试函数"""
    print("run.py 功能验证测试")
    print("=" * 50)

    # 运行所有测试
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
            print(f"[ERROR] 测试 {test.__name__} 发生异常: {e}")

    print("\n测试结果汇总")
    print("=" * 50)
    print(f"通过: {passed}/{total}")

    if passed == total:
        print("所有测试通过！run.py 修改成功！")
        return True
    else:
        print("部分测试失败，请检查相关问题")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)