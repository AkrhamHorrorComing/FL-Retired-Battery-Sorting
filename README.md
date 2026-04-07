# Second-Level Heterogeneous Retired Battery Chemistry Identification Using Pulse Test Enabled Federated Learning with Hard Privacy Guarantee

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Federated Learning](https://img.shields.io/badge/Federated%20Learning-PyTorch-orange.svg)](https://pytorch.org)

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Data Description](#data-description)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Algorithms](#core-algorithms)
- [Experiment Configuration](#experiment-configuration)
- [Output File Structure](#output-file-structure)
- [Experiment Results](#experiment-results)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview

This project implements a **Federated Learning (FL) system for retired battery chemistry classification**. It identifies the chemical composition and type of retired lithium-ion batteries using short-pulse test data, without sharing raw data across participating facilities. The system supports 8 distinct battery types spanning three chemistries (LMO, NMC, LFP), and provides hard differential-privacy guarantees through Laplace noise injection on statistical descriptors.

The federated approach is critical here because battery recyclers and second-life evaluators each hold proprietary datasets. By training local models per client and aggregating only predictions (or differentially-private statistics), no raw pulse-test waveform ever leaves the site.

**Supported Battery Types:**

| Index | Type | Chemistry |
|-------|------|-----------|
| 0 | 10Ah_LMO | Lithium Manganese Oxide |
| 1 | 15Ah_NMC | Nickel Manganese Cobalt |
| 2 | 21Ah_NMC | Nickel Manganese Cobalt |
| 3 | 24Ah_LMO | Lithium Manganese Oxide |
| 4 | 25Ah_LMO | Lithium Manganese Oxide |
| 5 | 26Ah_LMO | Lithium Manganese Oxide |
| 6 | 35Ah_LFP | Lithium Iron Phosphate |
| 7 | 68Ah_LFP | Lithium Iron Phosphate |

## Key Features

- **Federated Learning Architecture**: Distributed training across multiple clients with full data privacy preservation.
- **Differential Privacy (DP)**: Laplace mechanism applied to mean and covariance statistics, with configurable privacy budget (epsilon).
- **Multi-Battery Type Support**: Classification of 8 battery types across 3 chemistries (LMO, NMC, LFP).
- **Multiple ML Models**: MLP (4-hidden-layer and 2-hidden-layer variants), Random Forest, K-Nearest Neighbors, Gaussian Process, and Decision Tree.
- **Data Augmentation**: Two strategies -- simple oversampling to a fixed target count, and covariance-guided correlated noise injection.
- **Intelligent Aggregation Algorithms**:
  - **FedAvg (Average Aggregation)**: Type-count-weighted probability averaging.
  - **Voting Aggregation**: Confidence-thresholded (0.6) voting with type-count weighting.
  - **Mahalanobis Distance Weighting**: Softmax over negative Mahalanobis distances, using DP-protected statistics.
  - **Encoder Softmax Weighting**: Reconstruction-error-based weighting via per-type autoencoders with softmax.
  - **Encoder Argmin Selection**: Selecting the client with the lowest reconstruction error for each test sample.
- **Visualization and Analysis**: Pie charts, stacked bar plots, confusion matrices, similarity heatmaps, per-class accuracy plots, SOH/SOC distribution histograms.
- **Experiment Framework**: 100-repeat randomized experiments with automatic report generation.

## Project Structure

```
FL-Retired-Battery-Sorting/
├── central.py                  # Centralized (non-FL) baseline model training
├── client_model.py             # Client model class definition (training, prediction, encoder)
├── dataset.py                  # Dataset partitioning and client generation
├── run.py                      # Federated learning main runner script
├── reat_dataset.py             # Read split_dataset files and create train/test CSVs
├── split_dataset.py            # Split raw data into train/test at 8:2 ratio (100 seeds)
├── plot.py                     # Visualization utilities (pie, bar, heatmap, histogram)
├── dp_secret.py                # Differential privacy utilities (Laplace mechanism, PSD repair)
├── example_usage.py            # Interactive quick-start script
├── test_run.py                 # Validation test script
├── setup.py                    # Package setup configuration
├── requirements.txt            # Python dependencies
├── mat_data.csv                # **Raw battery pulse-test dataset (project root)**
├── README.md                   # This file
├── QUICK_START.md              # Quick-start guide
├── LICENSE                     # MIT License
├── exp/                        # Experiment utilities module
│   ├── encoder.py              # Autoencoder neural network (PyTorch)
│   └── evaluation.py           # Accuracy evaluation, confusion matrix, classification report
├── distance/                   # Aggregation and distance-computation modules
│   ├── aggregate_fed.py        # FedAvg aggregation (type-count weighted average)
│   ├── aggregate_1.py          # Voting aggregation (confidence-thresholded)
│   ├── Ma_distance1.py         # Mahalanobis distance-weighted aggregation (with DP)
│   ├── Ma_distance2.py         # Alternative Mahalanobis distance implementation
│   ├── novel_distance.py       # Encoder distance-weighted aggregation
│   ├── Encoder_distance.py     # Encoder distance utility
│   ├── Score_distance.py       # Score-based distance utility
│   └── Score_distance2.py      # Alternative score-based distance utility
├── split_dataset/              # Pre-computed train/test split indices (0.csv ~ 99.csv)
├── data/                       # Generated train/test CSV files (created at runtime)
├── client_model/               # Saved client models and per-experiment results (created at runtime)
├── centralized_model/          # Saved centralized models and results (created at runtime)
├── results/                    # Aggregated experiment result CSV files
└── plots/                      # Generated plot images
```

## Data Description

### Raw Data File Location

The primary dataset file is:

```
mat_data.csv
```

`mat_data.csv` is the raw data file for this project and must be placed in the **project root directory** (`FL-Retired-Battery-Sorting/mat_data.csv`). It is a tab-separated CSV file containing battery pulse-test measurements. This file is not tracked by git (see `.gitignore`); you must obtain it separately and place it in the project root before running any experiments.

### Dataset Columns

| Column | Range / Values | Description |
|--------|----------------|-------------|
| `No.` | Integer | Battery specimen serial number (unique per battery) |
| `condition` | 10Ah_LMO, 15Ah_NMC, 21Ah_NMC, 24Ah_LMO, 25Ah_LMO, 26Ah_LMO, 35Ah_LFP, 68Ah_LFP | Battery type label (classification target) |
| `SOCR` | 0.0 -- 1.0 | State of Charge (Ratio) at the time of the pulse test |
| `SOH` | 0.3 -- 1.0 | State of Health |
| `U1` -- `U41` | ~2.4 -- 4.4 V | 41 voltage features extracted from the pulse test waveform (input features) |

### Data Flow

```
mat_data.csv (project root)
        |
        v
  split_dataset.py  -->  split_dataset/{0..99}.csv  (train/test index lists)
        |
        v
  reat_dataset.py   -->  data/train{0..99}.csv, data/test{0..99}.csv
        |
        v
  dataset.py / run.py  -->  client_model/{seed}/client_{i}.pkl  (trained models)
```

1. `split_dataset.py` reads `mat_data.csv`, randomly splits each battery type into 80% train / 20% test (by battery ID), and saves the index lists to `split_dataset/{seed}.csv` for 100 random seeds.
2. `reat_dataset.py` reads the split indices and generates `data/train{seed}.csv` and `data/test{seed}.csv`.
3. `dataset.py` further partitions the training data across federated clients.

**Note:** The `split_dataset/` directory already contains 100 pre-computed split files (`0.csv` through `99.csv`). The `data/` directory is generated at runtime by running `split_dataset.py` or `reat_dataset.py`.

## Installation

### Prerequisites

- **Python**: 3.7 or higher
- **OS**: Windows, macOS, or Linux
- **Memory**: 4 GB+ recommended (8 GB for large-scale experiments)
- **Storage**: At least 2 GB free space

### Install Dependencies

Option 1 -- Using `pip` with `requirements.txt` (recommended):

```bash
pip install -r requirements.txt
```

Option 2 -- Manual installation:

```bash
pip install pandas>=1.3.0
pip install numpy>=1.20.0
pip install scikit-learn>=1.0.0
pip install torch>=1.9.0
pip install matplotlib>=3.3.0
pip install seaborn>=0.11.0
pip install scipy>=1.7.0
```

### Dependencies List

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | >= 1.3.0 | Data loading and manipulation |
| numpy | >= 1.20.0 | Numerical computation |
| scikit-learn | >= 1.0.0 | ML models (MLP, RF, KNN, GP, DT), LDA, metrics |
| torch | >= 1.9.0 | Autoencoder training |
| matplotlib | >= 3.3.0 | Plotting and visualization |
| seaborn | >= 0.11.0 | Statistical visualization (confusion matrices) |
| scipy | >= 1.7.0 | Distance computation, softmax |

## Quick Start

See [QUICK_START.md](QUICK_START.md) for a detailed 5-minute setup guide. Below is a summary:

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/FL-Retired-Battery-Sorting.git
cd FL-Retired-Battery-Sorting
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Prepare the Data

The raw data file `mat_data.csv` must be present in the project root directory. Then generate the train/test split files:

```bash
python split_dataset.py
```

This creates `split_dataset/{0..99}.csv` and `data/train{0..99}.csv`, `data/test{0..99}.csv`.

Alternatively, if `split_dataset/` files already exist:

```bash
python reat_dataset.py
```

### Step 4: Run Federated Learning Experiments

```bash
python run.py
```

This will:
1. Create necessary directories (`client_model/`, `data/`, `results/`, `plots/`).
2. Check that data files exist.
3. Ask whether to generate new client models.
4. Run 100 random experiments with 5 aggregation methods.
5. Save results to `results/federated_learning_results.csv`.
6. Generate visualization plots.

### Step 5: Run Centralized Baseline (Comparison)

```bash
python central.py
```

This trains a single centralized MLP on all training data (LDA + 4-hidden-layer MLP) for 100 random seeds, saving models and accuracy to `centralized_model/`.

### Step 6: Interactive Quick Start (Optional)

```bash
python example_usage.py
```

This provides an interactive menu with three modes: quick start example, advanced example, and full FL training.

## Core Algorithms

### Client Model (`client_model.py`)

The `client_model` class encapsulates a single federated client:

- **Data Augmentation Strategy 1** (`data_augmentation_1`): Simple random oversampling to reach a fixed target of 1600 samples per type.
- **Data Augmentation Strategy 2** (`data_augmentation_2`): Computes the covariance matrix of the feature space per type, then adds correlated multivariate Gaussian noise to augmented samples.
- **Feature Reduction**: Linear Discriminant Analysis (LDA) reduces the 41-dimensional input to `(num_types - 1)` dimensions.
- **Supported Models**:
  - `MLP` -- 4 hidden layers of 100 neurons each.
  - `MLP_2` -- 2 hidden layers of 50 neurons each.
  - `RF` -- Random Forest (unlimited depth).
  - `KNN` -- K-Nearest Neighbors (k=10, distance-weighted).
  - `GP` -- Gaussian Process (RationalQuadratic kernel).
  - `DT` -- Decision Tree (max depth 100).
- **Autoencoder** (`set_encoder`): For each battery type the client holds, a separate autoencoder (input -> 32 -> 16 -> hidden_dim -> 16 -> 32 -> input) is trained. Reconstruction error is used for out-of-distribution (OOD) detection during aggregation.

### Differential Privacy (`dp_secret.py`)

The DP module provides:

- `dp_mean_manual`: Laplace-noised mean computation with configurable epsilon.
- `dp_covariance_manual_robust`: Laplace-noised covariance with post-processing to ensure positive semi-definiteness (PSD) via eigenvalue clipping.
- `get_dp_stats_fully_manual_robust`: Splits the privacy budget (20% for mean, 80% for covariance) and returns both noised statistics.

Privacy budget is controlled by the `epsilon` parameter passed to `Ma_distance1.distance_weighted_test()`.

### Aggregation Strategies (`distance/`)

1. **Average Aggregation** (`aggregate_fed.py`):
   - Each client's probability output is weighted by `alpha = len(client.types) / total_types`.
   - The weighted probabilities are summed, and the argmax gives the final prediction.

2. **Voting Aggregation** (`aggregate_1.py`):
   - Probabilities below 0.6 confidence are zeroed out (hard-threshold filtering).
   - Remaining probabilities are weighted by `alpha` and summed.

3. **Mahalanobis Distance Weighting** (`Ma_distance1.py`):
   - Computes the Mahalanobis distance from each test point to each client's training data distribution (using DP-protected mean and covariance).
   - Distances are converted to weights via `softmax(-distance / temperature)`.
   - Client predictions are aggregated using these distance-based weights.

4. **Encoder Softmax Weighting** (`novel_distance.py`, `sum_weight=1`):
   - For each test sample, reconstruction error is computed using each client's autoencoders (minimum error across types).
   - Errors are converted to weights via `softmax(-error / 0.05)`.
   - Probability outputs are aggregated using these weights.

5. **Encoder Argmin Selection** (`novel_distance.py`, `sum_weight=0`):
   - The client with the lowest reconstruction error for each test sample is selected.
   - Only that client's prediction is used (hard selection).

### Dataset Partitioning (`dataset.py`)

The `partition_random_quantities` function distributes battery data across clients:

- Each client is randomly assigned `mini_type` to `max_type` battery types.
- For each type, batteries are randomly partitioned among the eligible clients, with a minimum of 5 batteries per client per type.
- If data is insufficient for all eligible clients, the number of clients is reduced.
- Ensures all data points are distributed (raises an error if any are missing).

### Centralized Baseline (`central.py`)

Trains a single model on all training data:
- LDA for dimensionality reduction.
- MLP with 4 hidden layers (100 neurons each).
- Runs for 100 random seeds.
- Saves per-seed model and accuracy to `centralized_model/`.

## Experiment Configuration

Key parameters can be modified at the top of `run.py`:

```python
MODEL_NAME = "MLP"          # Model type: "MLP", "MLP_2", "RF", "KNN", "GP", "DT"
NUM_CLIENT = 6              # Number of federated clients
NUM_EXPERIMENTS = 100       # Number of random experiment repetitions
MINI_TYPE = 2               # Minimum battery types per client
MAX_TYPE = 5                # Maximum battery types per client
HIDDEN_DIM = 10             # Autoencoder bottleneck dimension
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `"MLP"` | Machine learning model to use for each client |
| `NUM_CLIENT` | 6 | Number of participating clients in the federation |
| `NUM_EXPERIMENTS` | 100 | Number of random experiment repetitions |
| `MINI_TYPE` | 2 | Minimum number of battery types assigned to each client |
| `MAX_TYPE` | 5 | Maximum number of battery types assigned to each client |
| `HIDDEN_DIM` | 10 | Autoencoder bottleneck (latent) dimension |

The DP parameter `epsilon` is passed directly in `run.py` when calling `distance.Ma_distance1.distance_weighted_test(rd, NUM_CLIENT, epsilon=5)`.

## Output File Structure

After running experiments, the following directories and files are generated:

```
FL-Retired-Battery-Sorting/
├── data/                              # Generated train/test data
│   ├── train0.csv                     # Training data for seed 0
│   ├── test0.csv                      # Test data for seed 0
│   ├── train1.csv
│   ├── test1.csv
│   └── ...                            # Up to train99.csv / test99.csv
├── client_model/                      # Client models and per-experiment outputs
│   ├── 0/                             # Experiment seed 0
│   │   ├── client_0.pkl               # Trained client 0 model (pickle)
│   │   ├── client_1.pkl
│   │   ├── ...
│   │   ├── logging.txt                # Training and evaluation log
│   │   ├── similarity_heatmap.png     # Client similarity heatmap
│   │   ├── type_accuracy_horizontal.png # Per-type accuracy bar chart
│   │   ├── fig/                       # Per-client distribution plots
│   │   │   ├── client_0.png           # Data/battery pie chart for client 0
│   │   │   ├── ...
│   │   │   └── client_battery_distribution_stacked_bar.png
│   │   └── confusion_matrix.png       # Aggregated confusion matrix
│   ├── 1/                             # Experiment seed 1
│   └── ...
├── centralized_model/                 # Centralized baseline outputs
│   ├── central_model_0.pkl
│   ├── centralized_model_accuracy.csv
│   └── {seed}_confusion_matrix.png
├── results/                           # Aggregated results
│   └── federated_learning_results.csv # Summary of all experiments
└── plots/                             # Additional plots
```

### `federated_learning_results.csv` Columns

| Column | Description |
|--------|-------------|
| `rd` | Random seed / experiment index |
| `similarity` | Average pairwise client similarity score |
| `avg` | Accuracy with average aggregation |
| `voting` | Accuracy with voting aggregation |
| `ma_distance` | Accuracy with Mahalanobis distance weighting |
| `encoder_softmax` | Accuracy with encoder softmax weighting |
| `encoder_argmin` | Accuracy with encoder argmin selection |

## Experiment Results

### Performance Metrics

The system computes and reports:

- **Overall Accuracy**: Proportion of correctly classified test samples.
- **Per-Class Accuracy**: Accuracy for each of the 8 battery types.
- **Precision / Recall / F1-Score**: Full classification report via `sklearn.metrics.classification_report`.
- **Confusion Matrix**: Visual and textual confusion matrix saved per experiment.
- **Client Similarity**: Pairwise cosine similarity of normalized client data-distribution vectors.

### Aggregation Method Comparison

| Method | Description | Use Case |
|--------|-------------|----------|
| FedAvg | Type-count-weighted probability averaging | Baseline aggregation |
| Voting | Confidence-thresholded (>= 0.6) weighted voting | Ensemble of confident predictions |
| Ma_Distance | Mahalanobis distance-based softmax weighting with DP | Data-distribution-aware weighting |
| Encoder_Softmax | Autoencoder reconstruction error softmax weighting | OOD-aware soft weighting |
| Encoder_Argmin | Selecting the client with lowest reconstruction error | OOD-aware hard selection |

## Troubleshooting

### Common Issues

**1. Data File Not Found**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/test0.csv'
```
**Solution:** Run `python split_dataset.py` or `python reat_dataset.py` to generate the train/test data files. Ensure `mat_data.csv` exists in the project root.

**2. Raw Data File Missing**
```
FileNotFoundError: [Errno 2] No such file or directory: 'mat_data.csv'
```
**Solution:** Ensure `mat_data.csv` is placed in the project root directory (`FL-Retired-Battery-Sorting/mat_data.csv`).

**3. Out of Memory**
```
MemoryError during training
```
**Solution:** Reduce `NUM_EXPERIMENTS` (e.g., to 10) or `NUM_CLIENT` (e.g., to 4) in `run.py`.

**4. Module Import Error**
```
ModuleNotFoundError: No module named 'distance'
```
**Solution:** Ensure you are running scripts from the project root directory. The `distance/` and `exp/` directories must be on the Python path.

**5. Client Model File Missing**
```
Warning: Client model file not found: client_model/0/client_0.pkl
```
**Solution:** When prompted by `run.py`, select `y` to generate new client models, or manually call `generate_client_model()` from `dataset.py`.

**6. Singular Matrix in Mahalanobis Distance**
```
numpy.linalg.LinAlgError: Singular matrix
```
**Solution:** This is handled internally via regularization (1e-6 * I) and the PSD repair mechanism in `dp_secret.py`. If it persists, increase the regularization value.

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Validate Installation

Run the built-in test script:

```bash
python test_run.py
```

Expected output:
```
run.py Function Validation Test
==================================================
...
All tests passed! run.py modification successful!
```

## Performance Optimization

### Hardware

- **GPU Acceleration**: Install a CUDA-enabled PyTorch build. The autoencoder training in `client_model.py` will automatically use GPU if available.
- **Memory**: 8 GB+ RAM recommended for `NUM_EXPERIMENTS=100` with `NUM_CLIENT=6`.
- **Storage**: SSD recommended for faster file I/O when generating 100 experiment splits.

### Parameter Tuning for Faster Runs

```python
# Fast test configuration
NUM_CLIENT = 4              # Fewer clients
NUM_EXPERIMENTS = 10        # Fewer repetitions
HIDDEN_DIM = 5              # Smaller autoencoder bottleneck
MODEL_NAME = "MLP_2"        # Smaller MLP (2 layers x 50 neurons)
```

### Background Execution

```bash
# Linux / macOS
nohup python run.py > experiment.log 2>&1 &

# Windows (cmd)
start /B python run.py > experiment.log 2>&1

# Windows (PowerShell)
Start-Process python -ArgumentList "run.py" -NoNewWindow -RedirectStandardOutput "experiment.log"
```

## Contributing

Contributions are welcome. Please follow these steps:

### Reporting Issues

1. Check existing [Issues](https://github.com/yourusername/FL-Retired-Battery-Sorting/issues) for duplicates.
2. Open a new issue with a clear description, error log, and system environment details.

### Submitting Code

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push the branch: `git push origin feature/your-feature`
5. Open a Pull Request.

### Code Style

- Follow PEP 8 conventions.
- Add docstrings and comments where appropriate.
- Maintain backward compatibility.
- Include tests for new functionality.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

- **Email**: h-hu24@mails.tsinghua.edu.cn
