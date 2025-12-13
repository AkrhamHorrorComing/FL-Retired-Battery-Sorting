# (这是你提供的代码)
import client_model
import pickle
import pandas as pd
import numpy as np
import torch
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

def encoder_weighted_test(random_seed,num_client):
    client_list = []

    test_data = pd.read_csv("data/test" + str(random_seed) + ".csv", sep="\t")

    test_data = test_data[test_data.condition == "21Ah_NMC"]

    X_test = test_data.loc[:, "U1":"U41"]
    Y_test = test_data["condition"]


    print("✅ 1. 加载所有客户端模型...")
    for i in range(0, num_client):
        with open("client_model/" + str(random_seed) + "/client_" + str(i) + ".pkl", 'rb') as f:
            client = pickle.load(f)
            client_list.append(client)
    print("所有模型加载完毕。")

    prediction_list = []
    all_probs_list = []
    error_list = []
    for client in client_list:

        if Y_test.unique() in client.type:
            print(f"客户端 {client.client_id} 有这个型号")
        else:
            print(f"客户端 {client.client_id} 没有这个型号!!!!")



        encoder = client.encoder

        X_reconstruct = client.PCA.transform(X_test)
        # X_reconstruct = (X_test-2.2)/4.4
        reconstructed_data = encoder(torch.tensor(X_reconstruct, dtype=torch.float32))
        error = np.mean(np.abs(reconstructed_data.detach().numpy() - X_reconstruct), axis=1)
        error_list.append(error)

        prediction, prob_matrix = client.predict(X_test, if_decode=0)
        prob_matrix = np.where(prob_matrix > 0.6, prob_matrix, 0)
        all_probs_list.append(prob_matrix)
        prediction_list.append(prediction)

    all_probs_stacked = np.stack(all_probs_list, axis=1)
    # 误差矩阵堆叠，形状为 (样本数, 客户端数)
    all_errors_stacked = np.stack(error_list, axis=1)
    all_prediction_stacked = np.stack(prediction_list, axis=1)

    temperature = 1
    weights = softmax(-all_errors_stacked / temperature, axis=1)

    weights_reshaped = weights[:, :, np.newaxis]
    final_aggregated_probs = np.sum(all_probs_stacked * weights_reshaped, axis=1)
    print(f"Encoder聚合的权重矩阵: {weights_reshaped}")

    #相信重构误差最小的一个
    max_prob_index = np.argmin(all_errors_stacked, axis=1)
    num_samples = all_prediction_stacked.shape[0]
    row_indices = np.arange(num_samples)
    final_predictions_encoded = all_prediction_stacked[row_indices, max_prob_index]

    #概率加权
    # final_predictions_encoded = np.argmax(final_aggregated_probs, axis=1)


    Y_test_encoded = client_list[0].encode(list(Y_test))



    # 计算准确率
    final_accuracy = accuracy_score(Y_test_encoded, final_predictions_encoded)

    print("\n" + "=" * 50)
    with open(f"client_model/{random_seed}/logging.txt", 'a', encoding='utf-8') as f:
        print("Test result:",file=f)
    with open(f"client_model/{random_seed}/logging.txt", 'a', encoding='utf-8') as f:
        print("\n" + "=" * 50)
        print(f"🚀 基于Encoder min的最终准确率: {final_accuracy:.4f}",file=f)
    print(f"🚀 基于Encoder加权的联邦模型最终准确率: {final_accuracy:.4f}")
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

# encoder_weighted_test(0,6)