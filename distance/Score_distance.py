#基于PCA特征距离 计算最小距离 取最大特征的最小距离
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

    # test_data = test_data[test_data.condition == "10Ah_LMO"]

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
    score_list = []

    for client in client_list:

        prediction, prob_matrix = client.predict(X_test, if_decode=0)

        all_probs_list.append(prob_matrix)
        prediction_list.append(prediction)

        scores = client.calculate_pca_distance(X_test,distance = 'euclidean')

        score_list.append(scores)

    all_score_list = np.stack(score_list, axis=1)

    selected_index = np.argmin(all_score_list,axis=1)

    all_prediction_list = np.stack(prediction_list, axis=1)


    final_predictions_encoded = all_prediction_list[np.arange(selected_index.shape[0]), selected_index]

    #概率加权




    Y_test_encoded = client_list[0].encode(list(Y_test))



    # 计算准确率
    final_accuracy = accuracy_score(Y_test_encoded, final_predictions_encoded)

    print("\n" + "=" * 50)
    with open(f"client_model/{random_seed}/logging.txt", 'a', encoding='utf-8') as f:
        print("Test result:",file=f)
    with open(f"client_model/{random_seed}/logging.txt", 'a', encoding='utf-8') as f:
        print("\n" + "=" * 50)
        print(f"🚀 基于Score_min的最终准确率: {final_accuracy:.4f}",file=f)
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
