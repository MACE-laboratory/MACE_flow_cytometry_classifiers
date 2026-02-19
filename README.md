# Flow Cytometry Isolate Classifiers
### Author: Massimo Bourquin, February 2026

This project trains **pairwise ensemble classifiers** to distinguish between microbial isolates using flow cytometry measurements.
For every pair of isolates, the script builds an ensemble of the best-performing models and evaluates them on a held-out test set.

The pipeline includes:

* Automated data ingestion from multiple experiments
* Metadata integration
* Outlier removal using Isolation Forest
* Feature engineering and scaling
* Feature selection via point-biserial correlation
* Evaluation of multiple machine learning models
* Ensemble creation using the top-performing models
* Saving trained models and performance statistics

---

## Overview of the Workflow

The script performs the following steps:

1. **Load and combine data**

   * Reads all `.csv` files from a directory.
   * Matches each sample to isolate metadata.
   * Assigns isolate labels.

2. **Remove low-count isolates**

   * Filters isolates with insufficient events.

3. **Outlier detection**

   * Uses Isolation Forest independently for each isolate.

4. **Feature engineering**

   * Original features
   * Log-transformed features
   * Square-root transformed features

5. **Feature scaling**

   * Standardizes all features using `StandardScaler`.

6. **Balanced sampling**

   * Up to **12,000 events per isolate**
   * Split into:

     * 10,000 training
     * 2,000 testing

7. **Feature selection**

   * Uses **point-biserial correlation**
   * Top feature sets:

     * 10 features
     * 20 features
     * 30 features

8. **Model evaluation**
   Base models tested:

   * Random Forest (multiple depths)
   * k-Nearest Neighbors
   * Gaussian Naive Bayes
   * Neural Networks (MLP)
   * AdaBoost

9. **Ensemble construction**

   * Select the **top 5 models across all feature sets**
   * Combine predictions using majority voting
   * Average predicted probabilities

10. **Metrics computed**

    * Balanced accuracy
    * F1 score
    * ROC AUC

11. **Model saving**

    * Each pairwise classifier is saved separately.

---

## Directory Structure

After running the script, the output looks like:

```
project/
│
├── models/
│   ├── isolate_A_vs_isolate_B/
│   │   ├── RF_md8_top20.pkl
│   │   ├── RF_md8_top20_features.pkl
│   │   ├── class_mappings.pkl
│   │   ├── ensemble_metadata.pkl
│   │
│   └── isolate_A_vs_isolate_C/
│
├── all_pairwise_stats.csv
└── pairwise_training_script.py
```

---

## Input Data Requirements

### Flow Cytometry Data

The script expects:

```
data/
    16_*.csv
    29_*.csv
```

Each CSV should contain **numeric flow cytometry features** such as:

* FSC
* SSC
* Fluorescence channels
* Derived cytometry measurements

---

### Metadata Files

Two metadata tables are required:

```
16012026_FC_dilutions.csv
23012026_FC_dilutions.csv
```

They must include:

| Column  | Description       |
| ------- | ----------------- |
| sample  | sample identifier |
| isolate | isolate name      |

Example:

```
sample,isolate
001,12
002,12
003,34
```

---

## Configuration

Edit these variables in `main()`:

```python
metadata_path_16 = "path/to/metadata_16.csv"
metadata_path_29 = "path/to/metadata_29.csv"
data_dir = "path/to/data"
remove_list = ["isolate_58", "isolate_25"]
```

These control:

* Metadata sources
* Data directory
* Isolates removed from analysis

---

## Installation

Install the conda environment:

```bash
conda env create --name mlflow --file=mlflow.yaml
conda activate mlflow
```

---

## Running the Training

Run the script:

```bash
python pairwise_training_script.py
```

Output:

* Trained models for each isolate pair
* CSV with all performance statistics
* Console summary of results

---

## Output Metrics

For each isolate pair, the script reports:

```
Pair isolate_X vs isolate_Y
  Pair balanced accuracy
  Pair F1
  Pair AUC
```

And per-class performance:

* Balanced accuracy
* F1 score
* ROC AUC

This helps identify:

* Hard-to-distinguish isolates
* High-performing model combinations

---

## Model Ensemble Strategy

Each pairwise classifier is an ensemble of:

```
Top 5 models across:
- Multiple model types
- Multiple feature sets
```

Voting method:

* Majority vote on predictions
* Mean probability for AUC evaluation

This improves robustness compared to a single model.

---

## Feature Engineering Details

For each cytometry parameter:

The script generates:

```
original_feature
log_feature
sqrt_feature
```

Log features automatically handle negative values using offsets. All features are standardized before modeling.

---

## Statistical Output

The file:

```
all_pairwise_stats.csv
```

Contains:

| class_1 | class_2 | balanced_accuracy | f1_score | auc |
| ------- | ------- | ----------------- | -------- | --- |

This file is useful for:

* Downstream analysis
* Heatmaps
* Model comparison
* Identifying ambiguous isolates

---

## Reproducibility

Random seeds are fixed throughout the pipeline:

* Sampling
* Cross-validation
* Model initialization

This ensures reproducible training runs.

---

## Citation / Usage

If you use this pipeline in a publication or project, please cite the repository or mention the workflow for:

**Flow cytometry-based isolate classification using pairwise ensemble learning.**

---

