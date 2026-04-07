#!/usr/bin/env python3
"""
Federated Learning Battery Classification System - Main Runner Script

This script implements the complete federated learning workflow, including:
1. Generate client models
2. Run federated learning experiments
3. Evaluate performance of different aggregation methods
4. Generate experiment result reports
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

# Experiment configuration parameters
MODEL_NAME = "MLP"
NUM_CLIENT = 6
NUM_EXPERIMENTS = 100  # Run 100 random experiments
MINI_TYPE = 2  # Minimum battery types per client
MAX_TYPE = 5  # Maximum battery types per client
HIDDEN_DIM = 10  # Encoder hidden layer dimension

def setup_directories():
    """Create necessary directory structure"""
    directories = ["client_model", "data", "results", "plots"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("Directory structure created")

def check_data_files():
    """Check if required data files exist"""
    required_files = ["data/test0.csv"]
    missing_files = []

    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print(f"Missing required data files: {missing_files}")
        print("Please run 'python reat_dataset.py' first to generate data files")
        return False

    print("Data file check complete")
    return True

def generate_client_models_if_needed(rd):
    """Generate client models if they don't exist"""
    client_dir = f"client_model/{rd}"

    if not os.path.exists(client_dir):
        print(f"Generating client models for experiment {rd}...")
        generate_client_model(
            random_seed=rd,
            num_client=NUM_CLIENT,
            mini_type=MINI_TYPE,
            max_type=MAX_TYPE,
            model_type=MODEL_NAME,
            hidden_dim=HIDDEN_DIM
        )
        print(f"Client models for experiment {rd} generated")
    else:
        print(f"Client models for experiment {rd} already exist")

def load_client_models(rd):
    """Load all client models for the specified random seed"""
    client_model_list = []
    client_accuracies = {}

    # Load test data
    test_data = pd.read_csv(f"data/test{rd}.csv", sep="\t")
    X_test = test_data.loc[:, "U1":"U41"]
    Y_test = test_data["condition"]

    # Load all client models
    for i in range(NUM_CLIENT):
        client_path = f"client_model/{rd}/client_{i}.pkl"
        if os.path.exists(client_path):
            client = pickle.load(open(client_path, 'rb'))
            client_model_list.append(client)

            # Evaluate client performance on test set
            client.test_score(X_test, Y_test)
            class_accuracies = client.type_score(X_test, Y_test)
            client_accuracies[f'Client_{i}'] = class_accuracies
        else:
            print(f"Warning: Client model file not found: {client_path}")

    return client_model_list, client_accuracies, X_test, Y_test, test_data

def calculate_client_similarity(client_model_list):
    """Calculate similarity between clients (based on data distribution)"""
    similarity_scores = []
    model_i_list = []
    model_j_list = []

    for model_i, model_j in itertools.combinations(client_model_list, 2):
        model_i_list.append(model_i.client_id)
        model_j_list.append(model_j.client_id)
        # Calculate data distribution similarity
        similarity_score = np.dot(model_i.dataset_num_list, model_j.dataset_num_list.T)[0, 0]
        similarity_scores.append(similarity_score)

    return model_i_list, model_j_list, similarity_scores, np.mean(similarity_scores)

def run_aggregation_methods(rd, test_data):
    """Run different aggregation methods and return accuracy"""
    results = {}

    try:
        # 1. Average Aggregation
        from distance import aggregate_fed
        results['avg'] = aggregate_fed.aggregate_test(NUM_CLIENT, rd, test_data)
    except Exception as e:
        print(f"Average aggregation failed: {e}")
        results['avg'] = 0.0

    try:
        # 2. Voting Aggregation
        from distance import aggregate_1
        results['voting'] = aggregate_1.aggregate_test(NUM_CLIENT, rd, test_data)
    except Exception as e:
        print(f"Voting aggregation failed: {e}")
        results['voting'] = 0.0

    try:
        # 3. Mahalanobis distance-weighted aggregation
        import distance.Ma_distance1
        results['ma_distance'] = distance.Ma_distance1.distance_weighted_test(
            rd, NUM_CLIENT, epsilon=5
        )
    except Exception as e:
        print(f"Mahalanobis distance-weighted aggregation failed: {e}")
        results['ma_distance'] = 0.0

    try:
        # 4. Encoder-weighted aggregation (softmax weights)
        import distance.novel_distance
        results['encoder_softmax'] = distance.novel_distance.encoder_weighted_test(
            rd, NUM_CLIENT, sum_weight=1
        )
    except Exception as e:
        print(f"Encoder softmax-weighted aggregation failed: {e}")
        results['encoder_softmax'] = 0.0

    try:
        # 5. Encoder-weighted aggregation (argmax weights)
        import distance.novel_distance
        results['encoder_argmin'] = distance.novel_distance.encoder_weighted_test(
            rd, NUM_CLIENT, sum_weight=0
        )
    except Exception as e:
        print(f"Encoder argmax-weighted aggregation failed: {e}")
        results['encoder_argmin'] = 0.0

    return results

def run_single_experiment(rd):
    """Run a single federated learning experiment"""
    print(f"\nRunning experiment {rd}/{NUM_EXPERIMENTS}")
    print("-" * 50)

    # Generate client models (if needed)
    generate_client_models_if_needed(rd)

    # Load client models
    client_model_list, client_accuracies, X_test, Y_test, test_data = load_client_models(rd)

    if not client_model_list:
        print(f"Experiment {rd}: No valid client models found, skipping")
        return None

    print(f"Successfully loaded {len(client_model_list)} client models")

    # Calculate client similarity
    if len(client_model_list) >= 2:
        model_i_list, model_j_list, similarity_scores, avg_similarity = calculate_client_similarity(client_model_list)
        print(f"Average client similarity: {avg_similarity:.4f}")

        # Generate similarity heatmap
        plot.similarity_heatmap(
            model_i_list, model_j_list, similarity_scores,
            f"client_model/{rd}/similarity_heatmap"
        )
    else:
        avg_similarity = 0.0

    # Plot client type accuracy
    if client_accuracies:
        client_model_list[0].plot_type_accuracy(client_accuracies, save_path=f"client_model/{rd}/type_accuracy")

    # Run aggregation methods
    print("Running aggregation methods...")
    aggregation_results = run_aggregation_methods(rd, test_data)

    print("Experiment results:")
    for method, accuracy in aggregation_results.items():
        print(f"   {method}: {accuracy:.4f}")

    # Return results
    result = {
        'rd': rd,
        'similarity': avg_similarity,
        **aggregation_results
    }

    return result

def save_results(results_df, filename="results/federated_learning_results.csv"):
    """Save experiment results to CSV file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    results_df.to_csv(filename, index=False, sep="\t")
    print(f"Results saved to: {filename}")

def generate_summary_report(results_df):
    """Generate summary report"""
    if results_df.empty:
        print("No experiment results to summarize")
        return

    print("\nExperiment Summary Report")
    print("=" * 50)

    # Calculate mean accuracy and std for each method
    methods = ['avg', 'voting', 'ma_distance', 'encoder_softmax', 'encoder_argmin']

    for method in methods:
        if method in results_df.columns:
            mean_acc = results_df[method].mean()
            std_acc = results_df[method].std()
            print(f"{method:20}: {mean_acc:.4f} +/- {std_acc:.4f}")

    # Find the best method
    method_means = {}
    for method in methods:
        if method in results_df.columns:
            method_means[method] = results_df[method].mean()

    if method_means:
        best_method = max(method_means.items(), key=lambda x: x[1])
        print(f"\nBest aggregation method: {best_method[0]} (mean accuracy: {best_method[1]:.4f})")

def main():
    """Main function: run federated learning experiments"""
    print("Federated Learning Battery Classification System")
    print("=" * 50)
    print(f"Experiment configuration:")
    print(f"  Model type: {MODEL_NAME}")
    print(f"  Number of clients: {NUM_CLIENT}")
    print(f"  Number of experiments: {NUM_EXPERIMENTS}")
    print(f"  Battery type range: {MINI_TYPE}-{MAX_TYPE}")
    print(f"  Hidden layer dimension: {HIDDEN_DIM}")
    print("-" * 50)

    # Set up directory structure
    setup_directories()

    # Check data files
    if not check_data_files():
        return

    # Ask user whether to generate new client models
    user_input = input("\nGenerate new client models? (y/n): ").strip().lower()
    if user_input == 'y':
        print("Starting client model generation...")
        for rd in range(NUM_EXPERIMENTS):
            print(f"Generating client models for experiment {rd}...")
            generate_client_model(
                random_seed=rd,
                num_client=NUM_CLIENT,
                mini_type=MINI_TYPE,
                max_type=MAX_TYPE,
                model_type=MODEL_NAME,
                hidden_dim=HIDDEN_DIM
            )
        print("All client models generated")

    # Run experiments
    results = []

    try:
        for rd in range(NUM_EXPERIMENTS):
            result = run_single_experiment(rd)
            if result:
                results.append(result)

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")

    except Exception as e:
        print(f"\nError during experiment: {e}")

    # Save results
    if results:
        results_df = pd.DataFrame(results)
        save_results(results_df)
        generate_summary_report(results_df)

        # Generate visualization charts
        print("\nGenerating visualization charts...")
        try:
            # More visualization code can be added here
            plot.plot(num_client=NUM_CLIENT, random_seed=0)
            print("Visualization charts generated")
        except Exception as e:
            print(f"Visualization chart generation failed: {e}")

    print("\nFederated learning experiments complete!")
    print(f"Results saved at: results/federated_learning_results.csv")

if __name__ == "__main__":
    main()
