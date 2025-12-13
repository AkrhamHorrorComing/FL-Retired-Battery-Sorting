#!/usr/bin/env python3
"""
联邦学习电池分类系统 - 主运行脚本
Federated Learning Battery Classification System - Main Runner Script

该脚本实现了完整的联邦学习流程，包括：
1. 生成客户端模型
2. 运行联邦学习实验
3. 评估不同聚合方法的性能
4. 生成实验结果报告
"""

import os
import pickle
import pandas as pd
import numpy as np
import itertools
from pathlib import Path

import dataset
from dataset import generate_client_model
import plot

# 实验配置参数
MODEL_NAME = "MLP"
NUM_CLIENT = 6
NUM_EXPERIMENTS = 100  # 运行100次随机实验
MINI_TYPE = 2  # 每个客户端最少电池类型数
MAX_TYPE = 5  # 每个客户端最多电池类型数
HIDDEN_DIM = 10  # 编码器隐藏层维度

def setup_directories():
    """创建必要的目录结构"""
    directories = ["client_model", "data", "results", "plots"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✅ 目录结构创建完成")

def check_data_files():
    """检查必要的数据文件是否存在"""
    required_files = ["data/test0.csv"]
    missing_files = []

    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print(f"❌ 缺少必要的数据文件: {missing_files}")
        print("请先运行 'python reat_dataset.py' 生成数据文件")
        return False

    print("✅ 数据文件检查完成")
    return True

def generate_client_models_if_needed(rd):
    """如果客户端模型不存在，则生成它们"""
    client_dir = f"client_model/{rd}"

    if not os.path.exists(client_dir):
        print(f"🔄 为实验 {rd} 生成客户端模型...")
        generate_client_model(
            random_seed=rd,
            num_client=NUM_CLIENT,
            mini_type=MINI_TYPE,
            max_type=MAX_TYPE,
            model_type=MODEL_NAME,
            hidden_dim=HIDDEN_DIM
        )
        print(f"✅ 实验 {rd} 的客户端模型生成完成")
    else:
        print(f"✅ 实验 {rd} 的客户端模型已存在")

def load_client_models(rd):
    """加载指定随机种子的所有客户端模型"""
    client_model_list = []
    client_accuracies = {}

    # 加载测试数据
    test_data = pd.read_csv(f"data/test{rd}.csv", sep="\t")
    X_test = test_data.loc[:, "U1":"U41"]
    Y_test = test_data["condition"]

    # 加载所有客户端模型
    for i in range(NUM_CLIENT):
        client_path = f"client_model/{rd}/client_{i}.pkl"
        if os.path.exists(client_path):
            client = pickle.load(open(client_path, 'rb'))
            client_model_list.append(client)

            # 评估客户端在测试集上的性能
            client.test_score(X_test, Y_test)
            class_accuracies = client.type_score(X_test, Y_test)
            client_accuracies[f'Client_{i}'] = class_accuracies
        else:
            print(f"⚠️  警告: 客户端模型文件不存在: {client_path}")

    return client_model_list, client_accuracies, X_test, Y_test, test_data

def calculate_client_similarity(client_model_list):
    """计算客户端之间的相似度（基于数据分布）"""
    similarity_scores = []
    model_i_list = []
    model_j_list = []

    for model_i, model_j in itertools.combinations(client_model_list, 2):
        model_i_list.append(model_i.client_id)
        model_j_list.append(model_j.client_id)
        # 计算数据分布的相似度
        similarity_score = np.dot(model_i.dataset_num_list, model_j.dataset_num_list.T)[0, 0]
        similarity_scores.append(similarity_score)

    return model_i_list, model_j_list, similarity_scores, np.mean(similarity_scores)

def run_aggregation_methods(rd, test_data):
    """运行不同的聚合方法并返回准确率"""
    results = {}

    try:
        # 1. 平均聚合 (Average Aggregation)
        from distance import aggregate_fed
        results['avg'] = aggregate_fed.aggregate_test(NUM_CLIENT, rd, test_data)
    except Exception as e:
        print(f"⚠️  平均聚合失败: {e}")
        results['avg'] = 0.0

    try:
        # 2. 投票聚合
        from distance import aggregate_1
        results['voting'] = aggregate_1.aggregate_test(NUM_CLIENT, rd, test_data)
    except Exception as e:
        print(f"⚠️  投票聚合失败: {e}")
        results['voting'] = 0.0

    try:
        # 3. 马氏距离加权聚合
        import distance.Ma_distance1
        results['ma_distance'] = distance.Ma_distance1.distance_weighted_test(
            rd, NUM_CLIENT, epsilon=5
        )
    except Exception as e:
        print(f"⚠️  马氏距离加权聚合失败: {e}")
        results['ma_distance'] = 0.0

    try:
        # 4. 编码器加权聚合（softmax权重）
        import distance.novel_distance
        results['encoder_softmax'] = distance.novel_distance.encoder_weighted_test(
            rd, NUM_CLIENT, sum_weight=1
        )
    except Exception as e:
        print(f"⚠️  编码器softmax加权聚合失败: {e}")
        results['encoder_softmax'] = 0.0

    try:
        # 5. 编码器加权聚合（argmax权重）
        import distance.novel_distance
        results['encoder_argmin'] = distance.novel_distance.encoder_weighted_test(
            rd, NUM_CLIENT, sum_weight=0
        )
    except Exception as e:
        print(f"⚠️  编码器argmax加权聚合失败: {e}")
        results['encoder_argmin'] = 0.0

    return results

def run_single_experiment(rd):
    """运行单个联邦学习实验"""
    print(f"\n🎲 运行实验 {rd}/{NUM_EXPERIMENTS}")
    print("-" * 50)

    # 生成客户端模型（如果需要）
    generate_client_models_if_needed(rd)

    # 加载客户端模型
    client_model_list, client_accuracies, X_test, Y_test, test_data = load_client_models(rd)

    if not client_model_list:
        print(f"❌ 实验 {rd}: 没有找到有效的客户端模型，跳过")
        return None

    print(f"✅ 成功加载 {len(client_model_list)} 个客户端模型")

    # 计算客户端相似度
    if len(client_model_list) >= 2:
        model_i_list, model_j_list, similarity_scores, avg_similarity = calculate_client_similarity(client_model_list)
        print(f"📊 客户端平均相似度: {avg_similarity:.4f}")

        # 生成相似度热力图
        plot.similarity_heatmap(
            model_i_list, model_j_list, similarity_scores,
            f"client_model/{rd}/similarity_heatmap"
        )
    else:
        avg_similarity = 0.0

    # 绘制客户端类型准确率
    if client_accuracies:
        client_model_list[0].plot_type_accuracy(client_accuracies, save_path=f"client_model/{rd}/type_accuracy")

    # 运行聚合方法
    print("🔄 运行聚合方法...")
    aggregation_results = run_aggregation_methods(rd, test_data)

    print("📊 本实验结果:")
    for method, accuracy in aggregation_results.items():
        print(f"   {method}: {accuracy:.4f}")

    # 返回结果
    result = {
        'rd': rd,
        'similarity': avg_similarity,
        **aggregation_results
    }

    return result

def save_results(results_df, filename="results/federated_learning_results.csv"):
    """保存实验结果到CSV文件"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    results_df.to_csv(filename, index=False, sep="\t")
    print(f"✅ 结果已保存到: {filename}")

def generate_summary_report(results_df):
    """生成汇总报告"""
    if results_df.empty:
        print("❌ 没有实验结果可汇总")
        return

    print("\n📋 实验汇总报告")
    print("=" * 50)

    # 计算各方法的平均准确率和标准差
    methods = ['avg', 'voting', 'ma_distance', 'encoder_softmax', 'encoder_argmin']

    for method in methods:
        if method in results_df.columns:
            mean_acc = results_df[method].mean()
            std_acc = results_df[method].std()
            print(f"{method:20}: {mean_acc:.4f} ± {std_acc:.4f}")

    # 找到最佳方法
    method_means = {}
    for method in methods:
        if method in results_df.columns:
            method_means[method] = results_df[method].mean()

    if method_means:
        best_method = max(method_means.items(), key=lambda x: x[1])
        print(f"\n🏆 最佳聚合方法: {best_method[0]} (平均准确率: {best_method[1]:.4f})")

def main():
    """主函数：运行联邦学习实验"""
    print("🚀 联邦学习电池分类系统")
    print("=" * 50)
    print(f"实验配置:")
    print(f"  模型类型: {MODEL_NAME}")
    print(f"  客户端数量: {NUM_CLIENT}")
    print(f"  实验次数: {NUM_EXPERIMENTS}")
    print(f"  电池类型范围: {MINI_TYPE}-{MAX_TYPE}")
    print(f"  隐藏层维度: {HIDDEN_DIM}")
    print("-" * 50)

    # 设置目录结构
    setup_directories()

    # 检查数据文件
    if not check_data_files():
        return

    # 询问用户是否要生成新的客户端模型
    user_input = input("\n是否要生成新的客户端模型？(y/n): ").strip().lower()
    if user_input == 'y':
        print("🔄 开始生成客户端模型...")
        for rd in range(NUM_EXPERIMENTS):
            print(f"生成实验 {rd} 的客户端模型...")
            generate_client_model(
                random_seed=rd,
                num_client=NUM_CLIENT,
                mini_type=MINI_TYPE,
                max_type=MAX_TYPE,
                model_type=MODEL_NAME,
                hidden_dim=HIDDEN_DIM
            )
        print("✅ 所有客户端模型生成完成")

    # 运行实验
    results = []

    try:
        for rd in range(NUM_EXPERIMENTS):
            result = run_single_experiment(rd)
            if result:
                results.append(result)

    except KeyboardInterrupt:
        print("\n⚠️  实验被用户中断")

    except Exception as e:
        print(f"\n❌ 实验过程中发生错误: {e}")

    # 保存结果
    if results:
        results_df = pd.DataFrame(results)
        save_results(results_df)
        generate_summary_report(results_df)

        # 生成可视化图表
        print("\n📊 生成可视化图表...")
        try:
            # 这里可以添加更多的可视化代码
            plot.plot(num_client=NUM_CLIENT, random_seed=0)
            print("✅ 可视化图表生成完成")
        except Exception as e:
            print(f"⚠️  可视化图表生成失败: {e}")

    print("\n🎉 联邦学习实验完成！")
    print(f"📁 结果保存在: results/federated_learning_results.csv")

if __name__ == "__main__":
    main()






