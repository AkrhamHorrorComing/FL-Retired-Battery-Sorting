# Based on PCA feature distance, compute minimum distance, use distance-weighted probability
import client_model
import pickle
import pandas as pd
import numpy as np
import torch
# Assuming your dataset.py and plot.py are also in the path
# import dataset
# from dataset import generate_client_model
# import plot
from matplotlib import pyplot as plt
import os

# Import required libraries
from scipy.spatial.distance import cdist
from scipy.special import softmax
from sklearn.metrics import accuracy_score

def encoder_weighted_test(random_seed,num_client):
    client_list = []

    test_data = pd.read_csv("data/test" + str(random_seed) + ".csv", sep="\t")

    # test_data = test_data[test_data.condition == "10Ah_LMO"]

    X_test = test_data.loc[:, "U1":"U41"]
    Y_test = test_data["condition"]


    print("1. Loading all client models...")
    for i in range(0, num_client):
        with open("client_model/" + str(random_seed) + "/client_" + str(i) + ".pkl", 'rb') as f:
            client = pickle.load(f)
            client_list.append(client)
    print("All models loaded.")

    prediction_list = []
    probs_list = []
    score_list = []

    for client in client_list:

        prediction, prob_matrix = client.predict(X_test, if_decode=0)

        probs_list.append(prob_matrix)
        prediction_list.append(prediction)

        scores = client.calculate_pca_distance(X_test,distance = 'euclidean')
        score_list.append(scores)

    all_score_list = np.stack(score_list, axis=1)
    all_probs_list = np.stack(probs_list, axis=1)
    temperature = 0.05  # Softmax temperature parameter, adjustable
    # Smaller distance means larger weight, so apply Softmax to negative distances
    weights = softmax(-all_score_list / temperature, axis=1)

    weights_reshaped = weights[:, :, np.newaxis]

    final_aggregated_probs = np.sum(all_probs_list * weights_reshaped, axis=1)

    final_aggregated_probs = np.argmax(final_aggregated_probs, axis=1)



    Y_test_encoded = client_list[0].encode(list(Y_test))



    # Calculate accuracy
    final_accuracy = accuracy_score(Y_test_encoded, final_aggregated_probs)

    print("\n" + "=" * 50)
    with open(f"client_model/{random_seed}/logging.txt", 'a', encoding='utf-8') as f:
        print("Test result:",file=f)
    with open(f"client_model/{random_seed}/logging.txt", 'a', encoding='utf-8') as f:
        print("\n" + "=" * 50)
        print(f"Score-weighted final accuracy: {final_accuracy:.4f}",file=f)
    print("=" * 50)


    label = client_list[0].decode(list(final_aggregated_probs))
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
