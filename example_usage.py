#!/usr/bin/env python3
"""
Federated Learning Battery Classification System - Quick Start Example
"""

import dataset
from client_model import client_model
import plot

def quick_start_example():
    """Quick start example for the federated learning battery classification system"""

    print("🚀 Starting federated learning battery classification system quick example")
    print("=" * 50)

    # 1. Generate client models
    print("📊 Generating client models...")
    dataset.generate_client_model(
        random_seed=42,
        num_client=4,           # 4 clients
        mini_type=2,            # Minimum 2 battery types per client
        max_type=4,             # Maximum 4 battery types per client
        model_type="MLP",       # Using MLP model
        hidden_dim=8            # Encoder hidden layer dimension
    )

    # 2. Plot training dataset distribution
    print("📈 Plotting dataset distribution...")
    dataset.plot_train_dataset(random_seed=42)

    # 3. Plot client data distribution
    print("📊 Plotting client data distribution...")
    plot.plot(num_client=4, random_seed=42)

    print("✅ Quick example complete!")
    print("📁 Generated files:")
    print("   - client_model/42/: Client models and results")
    print("   - Data distribution charts and model performance plots")

def advanced_example():
    """More advanced example demonstrating different configuration options"""

    print("🔬 Advanced Example - Comparing Different Configurations")
    print("=" * 50)

    # Different random seeds
    seeds = [42, 123, 456]

    for seed in seeds:
        print(f"\n🎲 Processing random seed: {seed}")

        # Generate client models
        dataset.generate_client_model(
            random_seed=seed,
            num_client=6,
            mini_type=2,
            max_type=5,
            model_type="MLP_2",   # Using smaller MLP architecture
            hidden_dim=10
        )

        # Plot results
        plot.plot(num_client=6, random_seed=seed)

    print("\n🎯 Advanced example complete!")
    print("📊 Can compare results across different random seeds")

if __name__ == "__main__":
    print("Federated Learning Battery Classification System")
    print("\nSelect run mode:")
    print("1. Quick start example")
    print("2. Advanced example")
    print("3. Run federated learning training")

    choice = input("\nEnter your choice (1/2/3): ").strip()

    if choice == "1":
        quick_start_example()
    elif choice == "2":
        advanced_example()
    elif choice == "3":
        print("🏃 Running federated learning training...")
        import run  # Import main runner script
    else:
        print("❌ Invalid choice, running quick start example...")
        quick_start_example()