#!/usr/bin/env python3
"""
联邦学习电池分类系统 - 快速开始示例
Federated Learning Battery Classification System - Quick Start Example
"""

import dataset
from client_model import client_model
import plot

def quick_start_example():
    """快速开始联邦学习电池分类系统的示例"""

    print("🚀 开始联邦学习电池分类系统快速示例")
    print("=" * 50)

    # 1. 生成客户端模型
    print("📊 生成客户端模型...")
    dataset.generate_client_model(
        random_seed=42,
        num_client=4,           # 4个客户端
        mini_type=2,            # 每个客户端最少2种电池类型
        max_type=4,             # 每个客户端最多4种电池类型
        model_type="MLP",       # 使用MLP模型
        hidden_dim=8            # 编码器隐藏层维度
    )

    # 2. 绘制训练数据集分布
    print("📈 绘制数据集分布...")
    dataset.plot_train_dataset(random_seed=42)

    # 3. 绘制客户端数据分布
    print("📊 绘制客户端数据分布...")
    plot.plot(num_client=4, random_seed=42)

    print("✅ 快速示例完成！")
    print("📁 查看生成的文件:")
    print("   - client_model/42/: 客户端模型和结果")
    print("   - 数据分布图和模型性能图表")

def advanced_example():
    """更高级的示例，展示不同的配置选项"""

    print("🔬 高级示例 - 不同配置的比较")
    print("=" * 50)

    # 不同的随机种子
    seeds = [42, 123, 456]

    for seed in seeds:
        print(f"\n🎲 处理随机种子: {seed}")

        # 生成客户端模型
        dataset.generate_client_model(
            random_seed=seed,
            num_client=6,
            mini_type=2,
            max_type=5,
            model_type="MLP_2",   # 使用较小的MLP架构
            hidden_dim=10
        )

        # 绘制结果
        plot.plot(num_client=6, random_seed=seed)

    print("\n🎯 高级示例完成！")
    print("📊 可以比较不同随机种子下的结果")

if __name__ == "__main__":
    print("联邦学习电池分类系统")
    print("Federated Learning Battery Classification System")
    print("\n选择运行模式:")
    print("1. 快速开始示例")
    print("2. 高级示例")
    print("3. 运行联邦学习训练")

    choice = input("\n请输入选择 (1/2/3): ").strip()

    if choice == "1":
        quick_start_example()
    elif choice == "2":
        advanced_example()
    elif choice == "3":
        print("🏃 运行联邦学习训练...")
        import run  # 导入主运行脚本
    else:
        print("❌ 无效选择，运行快速开始示例...")
        quick_start_example()