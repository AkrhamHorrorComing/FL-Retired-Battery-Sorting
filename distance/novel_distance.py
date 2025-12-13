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

def encoder_weighted_test(random_seed,num_client,sum_weight = 1):
    client_list = []

    test_data = pd.read_csv("data/test" + str(random_seed) + ".csv", sep="\t")

    # test_data = test_data[test_data.condition == "15Ah_NMC"]

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
    error_mean_list = []




    for client in client_list:

        # if Y_test.unique() in client.type:
        #     print(f"客户端 {client.client_id} 有这个型号")
        # else:
        #     print(f"客户端 {client.client_id} 没有这个型号!!!!")
        prediction, prob_matrix = client.predict(X_test, if_decode=0)

        all_probs_list.append(prob_matrix)
        prediction_list.append(prediction)

        encoder_dict = client.encoder

        error_flag = np.full((len(Y_test),), np.inf)

        for key,value in encoder_dict.items():
            data = client.encoder_PCA.transform(X_test)
            data = client.scaler[key].transform(data)
            reconstructed_data = value(
                torch.tensor(data, dtype=torch.float32))
            error_mean = np.mean(np.abs(reconstructed_data.detach().numpy() - data), axis=1)
            # error_mean = np.abs((reconstructed_data.detach().numpy() - data)[:,0])
            error_flag = np.where(error_flag > error_mean, error_mean, error_flag)
        error_mean_list.append(error_flag)



    all_probs_stacked = np.stack(all_probs_list, axis=1)
    # 误差矩阵堆叠，形状为 (样本数, 客户端数)
    all_errors_mean_stacked = np.stack(error_mean_list, axis=1)
    weights = softmax(-all_errors_mean_stacked/0.05, axis=1)
    weights = weights[:, :, np.newaxis]

    if sum_weight == 1:
        final_aggregated_probs = np.sum(all_probs_stacked * weights, axis=1)
        final_predictions_encoded = np.argmax(final_aggregated_probs, axis=1)
        result = final_predictions_encoded
        label = client_list[0].decode(list(final_predictions_encoded))
        true = list(Y_test)
        from exp.evaluation import evalute_accuracy
        evalute_accuracy(result, true, label, random_seed, "encoder加权")
    else:
        selected_index = np.argmin(all_errors_mean_stacked, axis=1)
        all_prediction_list = np.stack(prediction_list, axis=1)
        final_predictions_encoded = all_prediction_list[np.arange(selected_index.shape[0]), selected_index]
        result = final_predictions_encoded
        label = client_list[0].decode(list(final_predictions_encoded))
        true = list(Y_test)
        from exp.evaluation import evalute_accuracy
        evalute_accuracy(result, true, label, random_seed, "encoder_min")




