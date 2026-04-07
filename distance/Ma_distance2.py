# (This is the provided code)
import client_model
import pickle
import pandas as pd
import numpy as np
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

def distance_weighted_test(random_seed,num_client,epsilon):
    client_list = []

    test_data = pd.read_csv("data/test" + str(random_seed) + ".csv", sep="\t")
    X_test = test_data.loc[:, "U1":"U41"]
    Y_test = test_data["condition"]


    print("1. Loading all client models...")
    for i in range(0, num_client):
        with open("client_model/" + str(random_seed) + "/client_" + str(i) + ".pkl", 'rb') as f:
            client = pickle.load(f)
            client_list.append(client)
    print("All models loaded.")

    # ------------------------------------------------------------------
    # Extension starts
    # ------------------------------------------------------------------

    ### Step 1: Pre-compute mean and inverse covariance matrix for each client
    print("\n2. Pre-computing client data distribution (mean and inverse covariance matrix)...")
    client_distributions = []
    feature_dim = X_test.shape[1]
    feature_bounds = [(2.4, 4.4)] * feature_dim

    regularization = 1e-6  # Add regularization term to prevent singular covariance matrix

    for i, client in enumerate(client_list):
        # Get training dataset from client object
        features = client.dataset.loc[:, "U1":"U41"].values

        # Calculate mean and covariance
        mean = np.mean(features, axis=0)
        cov = np.cov(features, rowvar=False)
        from dp_secret import get_dp_stats_fully_manual_robust
        mean, cov = get_dp_stats_fully_manual_robust(
            features,
            epsilon,
            feature_bounds
        )


        # Add regularization term to ensure matrix is invertible
        cov += np.eye(feature_dim) * regularization
        inv_cov = np.linalg.inv(cov)

        client_distributions.append({'mean': mean, 'inv_cov': inv_cov})
        print(f"Client {i} distribution parameters computed.")

    ### Step 2: Get probability predictions from all clients on test set
    print("\n3. Getting probability predictions from all clients...")
    all_probs_list = []
    for client in client_list:
        _, prob_matrix = client.predict(X_test, if_decode=0)
        # prob_matrix = np.where(prob_matrix > 0.6, prob_matrix, 0)
        all_probs_list.append(prob_matrix)

    # Stack probability list into a 3D array (clients, n_samples, n_classes)
    # Then transpose to (n_samples, n_clients, n_classes) for subsequent calculations
    all_probs_stacked = np.stack(all_probs_list, axis=0).transpose((1, 0, 2))
    print(f"Combined probability matrix shape: {all_probs_stacked.shape}")

    ### Step 3: Efficiently compute Mahalanobis distance
    print("\n4. Computing Mahalanobis distance from each test point to each client distribution...")
    X_test_values = X_test.values
    distance_matrix = np.zeros((X_test_values.shape[0], num_client))

    for i, dist_params in enumerate(client_distributions):
        # Use cdist to efficiently compute Mahalanobis distance for a batch of data to a distribution
        distances = cdist(X_test_values, [dist_params['mean']],
                          metric='mahalanobis', VI=dist_params['inv_cov'])
        distance_matrix[:, i] = distances.flatten()

    print(f"Mahalanobis distance matrix shape: {distance_matrix.shape}")

    ### Step 4: Convert distances to weights
    print("\n5. Converting distances to weights using Softmax...")
    temperature = 1.0  # Softmax temperature parameter, adjustable
    # Smaller distance means larger weight, so apply Softmax to negative distances
    weights = softmax(-distance_matrix / temperature, axis=1)
    print(f"Weight matrix shape: {weights.shape}")
    # Print first sample's weights as example
    print(f"First sample weight distribution: {np.round(weights[0], 4)}")

    ### Step 5: Weighted aggregation of probabilities
    print("\n6. Weighted aggregation of all model prediction probabilities...")
    # Expand weight matrix from (n_samples, n_clients) to (n_samples, n_clients, 1)
    # For broadcast multiplication with probability matrix of shape (n_samples, n_clients, n_classes)
    weights_reshaped = weights[:, :, np.newaxis]

    import novel_mask
    errors_mask_reshaped = novel_mask.encoder_mask(random_seed,num_client)
    weights_reshaped = weights_reshaped*errors_mask_reshaped

    # Execute weighted sum
    final_aggregated_probs = np.sum(all_probs_stacked * weights_reshaped, axis=1)
    print(f"Final aggregated probability matrix shape: {final_aggregated_probs.shape}")

    ### Step 6: Evaluate final accuracy
    print("\n7. Evaluating aggregated model's final accuracy...")
    # Get final predicted classes (encoded)
    final_predictions_encoded = np.argmax(final_aggregated_probs, axis=1)

    # Encode the true Y_test labels as well for comparison
    # Any client's encoder can be used since they are identical
    Y_test_encoded = client_list[0].encode(list(Y_test))

    # Calculate accuracy
    final_accuracy = accuracy_score(Y_test_encoded, final_predictions_encoded)

    print("\n" + "=" * 50)
    with open(f"client_model/{random_seed}/logging.txt", 'a', encoding='utf-8') as f:
        print("Test result:",file=f)
    with open(f"client_model/{random_seed}/logging.txt", 'a', encoding='utf-8') as f:
        print("\n" + "=" * 50)
        print(f"Mahalanobis distance + encoder weighted federated model final accuracy: {final_accuracy:.4f}",file=f)
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


    print("1. Loading all client models...")
    for i in range(0, num_client):
        with open("client_model/" + str(random_seed) + "/client_" + str(i) + ".pkl", 'rb') as f:
            client = pickle.load(f)
            client_list.append(client)
    print("All models loaded.")

    # ------------------------------------------------------------------
    # Extension starts
    # ------------------------------------------------------------------

    ### Step 1: Pre-compute mean and inverse covariance matrix for each client
    print("\n2. Pre-computing client data distribution (mean and inverse covariance matrix)...")
    client_distributions = []
    feature_dim = X_test.shape[1]
    regularization = 1e-6  # Add regularization term to prevent singular covariance matrix

    for i, client in enumerate(client_list):
        # Get training dataset from client object
        features = client.dataset.loc[:, "U1":"U41"].values

        # Calculate mean and covariance
        mean = np.mean(features, axis=0)
        cov = np.cov(features, rowvar=False)

        # Add regularization term to ensure matrix is invertible
        cov += np.eye(feature_dim) * regularization
        inv_cov = np.linalg.inv(cov)

        client_distributions.append({'mean': mean, 'inv_cov': inv_cov})
        print(f"Client {i} distribution parameters computed.")

    ### Step 2: Get probability predictions from all clients on test set
    print("\n3. Getting probability predictions from all clients...")
    all_probs_list = []
    for client in client_list:
        _, prob_matrix = client.predict(X_test, if_decode=0)
        prob_matrix = np.where(prob_matrix > 0.6, prob_matrix, 0)
        all_probs_list.append(prob_matrix)

    # Stack probability list into a 3D array (clients, n_samples, n_classes)
    # Then transpose to (n_samples, n_clients, n_classes) for subsequent calculations
    all_probs_stacked = np.stack(all_probs_list, axis=0).transpose((1, 0, 2))
    print(f"Combined probability matrix shape: {all_probs_stacked.shape}")

    ### Step 3: Efficiently compute Mahalanobis distance
    print("\n4. Computing Mahalanobis distance from each test point to each client distribution...")
    X_test_values = X_test.values
    distance_matrix = np.zeros((X_test_values.shape[0], num_client))

    for i, dist_params in enumerate(client_distributions):
        # Use cdist to efficiently compute Mahalanobis distance for a batch of data to a distribution
        distances = cdist(X_test_values, [dist_params['mean']],
                          metric='mahalanobis', VI=dist_params['inv_cov'])
        distance_matrix[:, i] = distances.flatten()

    print(f"Mahalanobis distance matrix shape: {distance_matrix.shape}")

    ### Step 4: Convert distances to weights
    print("\n5. Converting distances to weights using Softmax...")
    temperature = 1.0  # Softmax temperature parameter, adjustable
    # Smaller distance means larger weight, so apply Softmax to negative distances
    weights = softmax(-distance_matrix / temperature, axis=1)
    print(f"Weight matrix shape: {weights.shape}")
    # Print first sample's weights as example
    print(f"First sample weight distribution: {np.round(weights[0], 4)}")

    ### Step 5: Weighted aggregation of probabilities
    print("\n6. Weighted aggregation of all model prediction probabilities...")
    # Expand weight matrix from (n_samples, n_clients) to (n_samples, n_clients, 1)
    # For broadcast multiplication with probability matrix of shape (n_samples, n_clients, n_classes)
    weights_reshaped = weights[:, :, np.newaxis]

    # Execute weighted sum
    final_aggregated_probs = np.sum(all_probs_stacked * weights_reshaped, axis=1)
    print(f"Final aggregated probability matrix shape: {final_aggregated_probs.shape}")

    ### Step 6: Evaluate final accuracy
    print("\n7. Evaluating aggregated model's final accuracy...")
    # Get final predicted classes (encoded)
    final_predictions_encoded = np.argmax(final_aggregated_probs, axis=1)

    # Encode the true Y_test labels as well for comparison
    # Any client's encoder can be used since they are identical
    Y_test_encoded = client_list[0].encode(list(Y_test))

    # Calculate accuracy
    final_accuracy = accuracy_score(Y_test_encoded, final_predictions_encoded)

    print("\n" + "=" * 50)
    with open(f"client_model/{random_seed}/logging.txt", 'a', encoding='utf-8') as f:
        print("Valid result:",file=f)
    with open(f"client_model/{random_seed}/logging.txt", 'a', encoding='utf-8') as f:
        print("\n" + "=" * 50)
        print(f"Mahalanobis distance weighted federated model final accuracy: {final_accuracy:.4f}",file=f)
    print(f"Mahalanobis distance weighted federated model final accuracy: {final_accuracy:.4f}")
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
