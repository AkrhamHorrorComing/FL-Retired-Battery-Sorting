# (This is the provided code)
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

    test_data = test_data[test_data.condition == "21Ah_NMC"]

    X_test = test_data.loc[:, "U1":"U41"]
    Y_test = test_data["condition"]


    print("1. Loading all client models...")
    for i in range(0, num_client):
        with open("client_model/" + str(random_seed) + "/client_" + str(i) + ".pkl", 'rb') as f:
            client = pickle.load(f)
            client_list.append(client)
    print("All models loaded.")

    prediction_list = []
    all_probs_list = []
    error_list = []
    for client in client_list:

        if Y_test.unique() in client.type:
            print(f"Client {client.client_id} has this model type")
        else:
            print(f"Client {client.client_id} does NOT have this model type!!!!")



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
    # Error matrix stacked, shape is (n_samples, n_clients)
    all_errors_stacked = np.stack(error_list, axis=1)
    all_prediction_stacked = np.stack(prediction_list, axis=1)

    temperature = 1
    weights = softmax(-all_errors_stacked / temperature, axis=1)

    weights_reshaped = weights[:, :, np.newaxis]
    final_aggregated_probs = np.sum(all_probs_stacked * weights_reshaped, axis=1)
    print(f"Encoder aggregated weight matrix: {weights_reshaped}")

    # Trust the one with the smallest reconstruction error
    max_prob_index = np.argmin(all_errors_stacked, axis=1)
    num_samples = all_prediction_stacked.shape[0]
    row_indices = np.arange(num_samples)
    final_predictions_encoded = all_prediction_stacked[row_indices, max_prob_index]

    # Probability weighting
    # final_predictions_encoded = np.argmax(final_aggregated_probs, axis=1)


    Y_test_encoded = client_list[0].encode(list(Y_test))



    # Calculate accuracy
    final_accuracy = accuracy_score(Y_test_encoded, final_predictions_encoded)

    print("\n" + "=" * 50)
    with open(f"client_model/{random_seed}/logging.txt", 'a', encoding='utf-8') as f:
        print("Test result:",file=f)
    with open(f"client_model/{random_seed}/logging.txt", 'a', encoding='utf-8') as f:
        print("\n" + "=" * 50)
        print(f"Encoder min final accuracy: {final_accuracy:.4f}",file=f)
    print(f"Encoder weighted federated model final accuracy: {final_accuracy:.4f}")
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
