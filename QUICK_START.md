# Quick Start Guide

This guide will help you run the Federated Learning Battery Classification System within 5 minutes.

## Table of Contents

- [Prerequisites](#prerequisites)
- [5-Minute Quick Demo](#5-minute-quick-demo)
- [Step-by-Step Full Pipeline](#step-by-step-full-pipeline)
- [Verifying the Installation](#verifying-the-installation)
- [Checking Output Files](#checking-output-files)
- [Common Issues](#common-issues)
- [Next Steps](#next-steps)

---

## Prerequisites

- **Python 3.7+** installed on your system
- **mat_data.csv** file present in the project root directory (`FL-Retired-Battery-Sorting/mat_data.csv`)

Check your Python version:

```bash
python --version
```

## 5-Minute Quick Demo

### Step 1: Install Dependencies

```bash
# Option A: Quick install (recommended)
pip install pandas numpy scikit-learn torch matplotlib seaborn scipy

# Option B: Using requirements.txt
pip install -r requirements.txt
```

### Step 2: Generate Train/Test Data

The raw data file `mat_data.csv` (located in the project root) must be present. Generate the split files:

```bash
python split_dataset.py
```

This creates:
- `split_dataset/0.csv` through `split_dataset/99.csv` (train/test index lists)
- `data/train0.csv` through `data/train99.csv` (training data)
- `data/test0.csv` through `data/test99.csv` (test data)

If `split_dataset/` files already exist, you can skip the full split and just regenerate data files:

```bash
python reat_dataset.py
```

### Step 3: Run the Interactive Quick Start

```bash
python example_usage.py
```

Select option **1** (Quick start example) when prompted. This will:
- Generate 4 client models for random seed 42
- Train an MLP classifier and autoencoder for each client
- Create data distribution visualization charts
- Save results to `client_model/42/`

## Step-by-Step Full Pipeline

### Full Federated Learning Experiment

```bash
python run.py
```

When prompted, enter `y` to generate new client models. The script will:
1. Create directories (`client_model/`, `data/`, `results/`, `plots/`)
2. Verify that data files exist
3. Generate client models for 100 random seeds
4. Run 5 aggregation methods on each experiment
5. Save results to `results/federated_learning_results.csv`
6. Generate summary report and visualization plots

### Centralized Baseline (Comparison)

```bash
python central.py
```

Trains a single MLP on the full dataset (no federated split) for comparison. Results are saved to `centralized_model/`.

## Verifying the Installation

Run the built-in validation script:

```bash
python test_run.py
```

Expected output:

```
run.py Function Validation Test
==================================================
Testing module imports...
[OK] pandas imported successfully
[OK] numpy imported successfully
[OK] dataset module imported successfully
[OK] plot module imported successfully
[OK] Distance aggregation modules imported successfully

Checking directory structure...
[OK] client_model/ directory exists
[OK] data/ directory exists
[OK] exp/ directory exists
[OK] distance/ directory exists

Testing run.py imports...
[OK] run.py file exists
[OK] function setup_directories defined
[OK] function check_data_files defined
[OK] function generate_client_models_if_needed defined
[OK] function load_client_models defined
[OK] function run_aggregation_methods defined
[OK] function run_single_experiment defined
[OK] function main defined

Checking configuration parameters...
[OK] MODEL_NAME = MLP
[OK] NUM_CLIENT = 6
[OK] NUM_EXPERIMENTS = 100
[OK] MINI_TYPE = 2
[OK] MAX_TYPE = 5
[OK] HIDDEN_DIM = 10

Test Results Summary
==================================================
Passed: 4/4
All tests passed! run.py modification successful!
```

## Checking Output Files

After a successful run, you should see the following files and directories:

```
FL-Retired-Battery-Sorting/
├── data/                         # Train/test CSV files
│   ├── train0.csv
│   ├── test0.csv
│   └── ...
├── client_model/42/              # Client models for seed 42
│   ├── client_0.pkl              # Trained client 0 (pickle)
│   ├── client_1.pkl
│   ├── client_2.pkl
│   ├── client_3.pkl
│   ├── logging.txt               # Training and evaluation log
│   ├── fig/                      # Data distribution plots
│   │   ├── client_0.png
│   │   ├── client_1.png
│   │   ├── client_2.png
│   │   └── client_3.png
│   └── ...
├── split_dataset/                # Pre-computed split indices
│   ├── 0.csv
│   ├── 1.csv
│   └── ...
└── results/                      # Experiment results (after full run)
    └── federated_learning_results.csv
```

### Key Files to Check

| File | Description |
|------|-------------|
| `client_model/42/logging.txt` | Detailed training and evaluation log for seed 42 |
| `client_model/42/fig/client_0.png` | Pie charts showing data and battery distribution for client 0 |
| `client_model/42/client_0.pkl` | Serialized client model (includes trained MLP, LDA, autoencoders) |
| `results/federated_learning_results.csv` | Aggregated accuracy across all experiments and methods |

## Common Issues

### 1. Raw Data File Missing

```
FileNotFoundError: mat_data.csv not found
```

**Solution:** Make sure `mat_data.csv` is in the project root directory:
```
FL-Retired-Battery-Sorting/mat_data.csv
```

This file is approximately 3.4 MB and contains the raw battery pulse-test dataset. It is not tracked by git (see `.gitignore`). You must obtain it separately and place it in the project root.

### 2. Train/Test Data Files Missing

```
FileNotFoundError: data/test0.csv not found
```

**Solution:** Generate the data files:
```bash
python split_dataset.py
```

### 3. Out of Memory

```
MemoryError during training
```

**Solution:** Reduce the experiment scale. Edit the configuration at the top of `run.py`:
```python
NUM_EXPERIMENTS = 10    # Reduce from 100 to 10
NUM_CLIENT = 4          # Reduce from 6 to 4
```

### 4. Module Import Error

```
ModuleNotFoundError: No module named 'dataset'
```

**Solution:** Run scripts from the project root directory:
```bash
cd FL-Retired-Battery-Sorting
python run.py
```

### 5. PyTorch Installation Issues

If you have difficulty installing PyTorch, visit [pytorch.org](https://pytorch.org/get-started/locally/) for OS-specific installation commands. PyTorch is only required for autoencoder training (encoder-based aggregation methods). The other aggregation methods (FedAvg, Voting, Mahalanobis) do not require PyTorch.

## Next Steps

Once you have successfully run the quick demo:

1. **Read the full documentation**: See [README.md](README.md) for comprehensive documentation on all features, configuration options, and aggregation algorithms.

2. **Run a full experiment**:
   ```bash
   python run.py
   ```
   This runs 100 random experiments with all 5 aggregation methods.

3. **Experiment with different models**: Edit `run.py` to change `MODEL_NAME` to `"RF"`, `"KNN"`, `"GP"`, or `"DT"` and compare results.

4. **Adjust the privacy budget**: Change the `epsilon` parameter in `run.py` (in the `run_aggregation_methods` function) to control the differential privacy strength. Lower epsilon means stronger privacy but more noise.

5. **Visualize results**: Use `plot.py` functions to generate custom visualizations:
   ```python
   import plot
   plot.plot(num_client=6, random_seed=0)
   plot.plot_stacked_bar_distribution(num_client=6, random_seed=0)
   ```

6. **Run the centralized baseline for comparison**:
   ```bash
   python central.py
   ```

## Getting Help

- Full documentation: [README.md](README.md)
- Report issues: [GitHub Issues](https://github.com/yourusername/FL-Retired-Battery-Sorting/issues)
- Contact: h-hu24@mails.tsinghua.edu.cn
