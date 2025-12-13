# (这是你提供的代码)
import client_model
import pickle
import pandas as pd
import numpy as np
# 假设你的 dataset.py 和 plot.py 也在路径中
# import dataset
# from dataset import generate_client_model
# import plot
from matplotlib import pyplot as plt
import os

# 导入后续需要的库
from scipy.spatial.distance import cdist
from scipy.special import softmax
from sklearn.metrics import accuracy_score

def distance_weighted_test(random_seed,num_client,epsilon):
    client_list = []

    test_data = pd.read_csv("data/test" + str(random_seed) + ".csv", sep="\t")
    X_test = test_data.loc[:, "U1":"U41"]
    Y_test = test_data["condition"]


    print("✅ 1. 加载所有客户端模型...")
    for i in range(0, num_client):
        with open("client_model/" + str(random_seed) + "/client_" + str(i) + ".pkl", 'rb') as f:
            client = pickle.load(f)
            client_list.append(client)
    print("所有模型加载完毕。")

    # ------------------------------------------------------------------
    # 续写部分开始
    # ------------------------------------------------------------------

    ### 步骤 1: 预计算每个客户端的均值和逆协方差矩阵
    print("\n✅ 2. 预计算客户端数据分布 (均值和逆协方差矩阵)...")
    client_distributions = []
    feature_dim = X_test.shape[1]
    feature_bounds = [(2.4, 4.4)] * feature_dim

    regularization = 1e-6  # 添加正则化项防止协方差矩阵奇异

    for i, client in enumerate(client_list):
        # 从客户端对象中获取其训练数据集
        features = client.dataset.loc[:, "U1":"U41"].values

        # 计算均值和协方差
        mean = np.mean(features, axis=0)
        cov = np.cov(features, rowvar=False)
        from dp_secret import get_dp_stats_fully_manual_robust
        mean, cov = get_dp_stats_fully_manual_robust(
            features,
            epsilon,
            feature_bounds
        )


        # 添加正则化项以确保矩阵可逆
        cov += np.eye(feature_dim) * regularization
        inv_cov = np.linalg.inv(cov)

        client_distributions.append({'mean': mean, 'inv_cov': inv_cov})
        print(f"客户端 {i} 的分布参数计算完成。")

    ### 步骤 2: 获取所有客户端对测试集的概率预测
    print("\n✅ 3. 从所有客户端获取概率预测...")
    all_probs_list = []
    for client in client_list:
        _, prob_matrix = client.predict(X_test, if_decode=0)
        # prob_matrix = np.where(prob_matrix > 0.6, prob_matrix, 0)
        all_probs_list.append(prob_matrix)

    # 将概率列表堆叠成一个三维数组 (客户端, 样本数, 类别数)
    # 然后转置为 (样本数, 客户端, 类别数) 以便后续计算
    all_probs_stacked = np.stack(all_probs_list, axis=0).transpose((1, 0, 2))
    print(f"组合后的概率矩阵形状: {all_probs_stacked.shape}")

    ### 步骤 3: 高效计算马氏距离
    print("\n✅ 4. 计算每个测试点到各客户端分布的马氏距离...")
    X_test_values = X_test.values
    distance_matrix = np.zeros((X_test_values.shape[0], num_client))

    for i, dist_params in enumerate(client_distributions):
        # 使用 cdist 高效计算一批数据到一个分布的马氏距离
        distances = cdist(X_test_values, [dist_params['mean']],
                          metric='mahalanobis', VI=dist_params['inv_cov'])
        distance_matrix[:, i] = distances.flatten()

    print(f"马氏距离矩阵形状: {distance_matrix.shape}")

    ### 步骤 4: 将距离转换为权重
    print("\n✅ 5. 使用 Softmax 将距离转换为权重...")
    temperature = 1.0  # Softmax 温度参数，可调
    # 距离越小，权重应越大，因此对负距离应用Softmax
    weights = softmax(-distance_matrix / temperature, axis=1)
    print(f"权重矩阵形状: {weights.shape}")
    # 打印第一个样本的权重作为示例
    print(f"第一个样本的权重分布: {np.round(weights[0], 4)}")

    ### 步骤 5: 加权聚合概率
    print("\n✅ 6. 加权聚合所有模型的预测概率...")
    # 将权重矩阵从 (n_samples, n_clients) 扩展为 (n_samples, n_clients, 1)
    # 以便与 (n_samples, n_clients, n_classes) 的概率矩阵进行广播乘法
    weights_reshaped = weights[:, :, np.newaxis]

    import novel_mask
    errors_mask_reshaped = novel_mask.encoder_mask(random_seed,num_client)
    weights_reshaped = weights_reshaped*errors_mask_reshaped

    # 执行加权求和
    final_aggregated_probs = np.sum(all_probs_stacked * weights_reshaped, axis=1)
    print(f"最终聚合后的概率矩阵形状: {final_aggregated_probs.shape}")

    ### 步骤 6: 评估最终准确率
    print("\n✅ 7. 评估聚合模型的最终准确率...")
    # 获取最终预测的类别（编码状态）
    final_predictions_encoded = np.argmax(final_aggregated_probs, axis=1)

    # 将真实的Y_test标签也进行编码，以便比较
    # 使用任意一个客户端的编码器即可，因为它们是相同的
    Y_test_encoded = client_list[0].encode(list(Y_test))

    # 计算准确率
    final_accuracy = accuracy_score(Y_test_encoded, final_predictions_encoded)

    print("\n" + "=" * 50)
    with open(f"client_model/{random_seed}/logging.txt", 'a', encoding='utf-8') as f:
        print("Test result:",file=f)
    with open(f"client_model/{random_seed}/logging.txt", 'a', encoding='utf-8') as f:
        print("\n" + "=" * 50)
        print(f"🚀 基于马氏距离+encoder加权的联邦模型最终准确率: {final_accuracy:.4f}",file=f)
    print("=" * 50)


    label = client_list[0].decode(list(final_predictions_encoded))
    true=list(Y_test)
    dict = {}
    for i in np.unique(true):
        class_indices = np.where(np.array(true) == i)[0]
        class_correct = 0
        dict[i] = []
        for j in class_indices:
            if true[j] == label[j]:
                class_correct += 1
            else:
                dict[i].append(label[j])
        print(f"Class {i} Accuracy: {class_correct/len(class_indices)}")
        with open(f"client_model/{random_seed}/logging.txt", 'a', encoding='utf-8') as f:
            print(f"Class {i} Accuracy: {class_correct/len(class_indices)}",file=f)
    return final_accuracy

def distance_weighted_valid(random_seed,num_client):
    client_list = []

    valid_data = pd.read_csv("data/valid" + str(random_seed) + ".csv", sep="\t")
    X_test = valid_data.loc[:, "U1":"U41"]
    Y_test = valid_data["condition"]


    print("✅ 1. 加载所有客户端模型...")
    for i in range(0, num_client):
        with open("client_model/" + str(random_seed) + "/client_" + str(i) + ".pkl", 'rb') as f:
            client = pickle.load(f)
            client_list.append(client)
    print("所有模型加载完毕。")

    # ------------------------------------------------------------------
    # 续写部分开始
    # ------------------------------------------------------------------

    ### 步骤 1: 预计算每个客户端的均值和逆协方差矩阵
    print("\n✅ 2. 预计算客户端数据分布 (均值和逆协方差矩阵)...")
    client_distributions = []
    feature_dim = X_test.shape[1]
    regularization = 1e-6  # 添加正则化项防止协方差矩阵奇异

    for i, client in enumerate(client_list):
        # 从客户端对象中获取其训练数据集
        features = client.dataset.loc[:, "U1":"U41"].values

        # 计算均值和协方差
        mean = np.mean(features, axis=0)
        cov = np.cov(features, rowvar=False)

        # 添加正则化项以确保矩阵可逆
        cov += np.eye(feature_dim) * regularization
        inv_cov = np.linalg.inv(cov)

        client_distributions.append({'mean': mean, 'inv_cov': inv_cov})
        print(f"客户端 {i} 的分布参数计算完成。")

    ### 步骤 2: 获取所有客户端对测试集的概率预测
    print("\n✅ 3. 从所有客户端获取概率预测...")
    all_probs_list = []
    for client in client_list:
        _, prob_matrix = client.predict(X_test, if_decode=0)
        prob_matrix = np.where(prob_matrix > 0.6, prob_matrix, 0)
        all_probs_list.append(prob_matrix)

    # 将概率列表堆叠成一个三维数组 (客户端, 样本数, 类别数)
    # 然后转置为 (样本数, 客户端, 类别数) 以便后续计算
    all_probs_stacked = np.stack(all_probs_list, axis=0).transpose((1, 0, 2))
    print(f"组合后的概率矩阵形状: {all_probs_stacked.shape}")

    ### 步骤 3: 高效计算马氏距离
    print("\n✅ 4. 计算每个测试点到各客户端分布的马氏距离...")
    X_test_values = X_test.values
    distance_matrix = np.zeros((X_test_values.shape[0], num_client))

    for i, dist_params in enumerate(client_distributions):
        # 使用 cdist 高效计算一批数据到一个分布的马氏距离
        distances = cdist(X_test_values, [dist_params['mean']],
                          metric='mahalanobis', VI=dist_params['inv_cov'])
        distance_matrix[:, i] = distances.flatten()

    print(f"马氏距离矩阵形状: {distance_matrix.shape}")

    ### 步骤 4: 将距离转换为权重
    print("\n✅ 5. 使用 Softmax 将距离转换为权重...")
    temperature = 1.0  # Softmax 温度参数，可调
    # 距离越小，权重应越大，因此对负距离应用Softmax
    weights = softmax(-distance_matrix / temperature, axis=1)
    print(f"权重矩阵形状: {weights.shape}")
    # 打印第一个样本的权重作为示例
    print(f"第一个样本的权重分布: {np.round(weights[0], 4)}")

    ### 步骤 5: 加权聚合概率
    print("\n✅ 6. 加权聚合所有模型的预测概率...")
    # 将权重矩阵从 (n_samples, n_clients) 扩展为 (n_samples, n_clients, 1)
    # 以便与 (n_samples, n_clients, n_classes) 的概率矩阵进行广播乘法
    weights_reshaped = weights[:, :, np.newaxis]

    # 执行加权求和
    final_aggregated_probs = np.sum(all_probs_stacked * weights_reshaped, axis=1)
    print(f"最终聚合后的概率矩阵形状: {final_aggregated_probs.shape}")

    ### 步骤 6: 评估最终准确率
    print("\n✅ 7. 评估聚合模型的最终准确率...")
    # 获取最终预测的类别（编码状态）
    final_predictions_encoded = np.argmax(final_aggregated_probs, axis=1)

    # 将真实的Y_test标签也进行编码，以便比较
    # 使用任意一个客户端的编码器即可，因为它们是相同的
    Y_test_encoded = client_list[0].encode(list(Y_test))

    # 计算准确率
    final_accuracy = accuracy_score(Y_test_encoded, final_predictions_encoded)

    print("\n" + "=" * 50)
    with open(f"client_model/{random_seed}/logging.txt", 'a', encoding='utf-8') as f:
        print("Valid result:",file=f)
    with open(f"client_model/{random_seed}/logging.txt", 'a', encoding='utf-8') as f:
        print("\n" + "=" * 50)
        print(f"🚀 基于马氏距离加权的联邦模型最终准确率: {final_accuracy:.4f}",file=f)
    print(f"🚀 基于马氏距离加权的联邦模型最终准确率: {final_accuracy:.4f}")
    print("=" * 50)


    label = client_list[0].decode(list(final_predictions_encoded))
    true=list(Y_test)
    dict = {}
    for i in np.unique(true):
        class_indices = np.where(np.array(true) == i)[0]
        class_correct = 0
        dict[i] = []
        for j in class_indices:
            if true[j] == label[j]:
                class_correct += 1
            else:
                dict[i].append(label[j])
        print(f"Class {i} Accuracy: {class_correct/len(class_indices)}")
        with open(f"client_model/{random_seed}/logging.txt", 'a', encoding='utf-8') as f:
            print(f"Class {i} Accuracy: {class_correct/len(class_indices)}",file=f)
    return final_accuracy
