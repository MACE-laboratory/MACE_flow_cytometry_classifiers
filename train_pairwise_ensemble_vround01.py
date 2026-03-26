import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import glob
import re

# =========================================================
# DATA LOADING AND PREPARATION FUNCTIONS
# =========================================================

import os
import re
import pandas as pd

import os
import re
import pandas as pd

def load_and_combine_data():
    """
    Loads all per-well monoculture CSVs in all ../data/round01/XXX2026/ subfolders.
    Returns a single dataframe concatenating all monocultures data with additional columns,
    and applies one-hot encoding to Temp, Day, isolate (isolate name for each row).
    """
    import os
    import re
    import pandas as pd

    # Load metadata (mono & pair info)
    metadata = pd.read_csv("../data/round01/all_metadata.csv")

    # File pattern for per-well csvs
    round01_dir = "../data/round01"
    filename_re = re.compile(
        r"(round\d+)_(\d+C)_(d\d+)_([0-9.]+)_([A-H][0-9]+)_([^\.]+)\.csv"
    )

    # Load pairs_to_wells.csv (in round01 dir)
    pairs_to_wells = pd.read_csv("../data/round01/pairs_to_wells.csv")

    # Make sure Day in pairs_to_wells is formatted as "dXX" (e.g. "d03")
    pairs_to_wells['Day'] = pairs_to_wells['Day'].astype(str).str.zfill(2)
    pairs_to_wells['Day'] = 'd' + pairs_to_wells['Day'].str[-2:]

    # Find all monoculture rows in the metadata
    mono_meta = metadata[metadata["Type"] == "mono"]
    # Map Community to Isolate for convenience
    community_to_isolate = dict(zip(mono_meta["Community"], mono_meta["IsolateA"]))

    # Crawl all round01/*2026/ folders and collect CSVs
    csv_records = []
    for folder in os.listdir(round01_dir):
        if not folder.endswith("2026"):
            continue
        folder_path = os.path.join(round01_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith('.csv'):
                continue
            match = filename_re.match(fname)
            if not match:
                continue
            round_n, temp, day, date, well, sample = match.groups()
            csv_records.append({
                "filepath": os.path.join(folder_path, fname),
                "filename": fname,
                "Round": round_n,
                "Temp": temp,
                "Day": day,
                "Date": date,
                "Well": well,
                "Sample": sample,
                "Folder": folder,
            })

    # Assign Community ID to each csv_record using Well, Day, Round from pairs_to_wells
    for rec in csv_records:
        day_str = rec["Day"]   # e.g., "d03"
        well = rec["Well"]
        round_val = rec["Round"]
        matches = pairs_to_wells[
            (pairs_to_wells["Well"] == well) &
            (pairs_to_wells["Day"] == day_str) &
            (pairs_to_wells["Round"] == round_val)
        ]["Community"].unique()
        if len(matches) == 1:
            rec["Community"] = matches[0]
        elif len(matches) == 0:
            rec["Community"] = None
            print(f"Warning: No Community match for Well={well}, Day={day_str}, Round={round_val}")
        else:
            raise ValueError(f"Multiple communities found for Well={well}, Day={day_str}, Round={round_val}: {matches}")

    # Filter to only monoculture csvs (Community in community_to_isolate.keys())
    monoculture_csvs = [rec for rec in csv_records if rec["Community"] in community_to_isolate]

    # Load, annotate, and collect all monoculture data
    dfs = []
    for rec in monoculture_csvs:
        df = pd.read_csv(rec["filepath"])
        isolate_label = community_to_isolate[rec["Community"]]
        sample_id = rec["Sample"]
        filename = rec["filename"]
        # Add key columns
        df["Temp"] = rec["Temp"]
        df["Day"] = rec["Day"]
        df["Date"] = rec["Date"]
        df["Well"] = rec["Well"]
        df["filename"] = filename
        df["isolate"] = isolate_label
        df["sample"] = sample_id
        dfs.append(df)

    # Concatenate all
    if not dfs:
        raise RuntimeError("No monoculture data found matching the metadata and pairs_to_wells mapping.")
    combined_df = pd.concat(dfs, ignore_index=True)

    # One-hot encode categorical columns
    categorical_cols = ["Temp", "Day"]
    combined_df = pd.get_dummies(combined_df, columns=categorical_cols)

    return combined_df

def remove_low_count_isolates(df, remove_list):
    """Remove isolates with low counts."""
    df_filtered = df[~df['isolate'].isin(remove_list)]
    print(f"Removed isolates: {remove_list}")
    print(f"Shape after removal: {df_filtered.shape}")
    return df_filtered


def clean_with_isolation_forest(df, numeric_cols, contamination=0.05, random_state=42):
    """Clean data using Isolation Forest outlier detection per isolate."""
    from sklearn.ensemble import IsolationForest
    
    cleaned_dfs = []

    for isolate, group in df.groupby('isolate'):
        iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
        iso_forest.fit(group[numeric_cols])
        
        inliers = iso_forest.predict(group[numeric_cols]) == 1
        cleaned_group = group[inliers]
        cleaned_dfs.append(cleaned_group)

    combined_clean = pd.concat(cleaned_dfs).reset_index(drop=True).dropna()
    print(f"Original rows: {len(df)}, Cleaned rows: {len(combined_clean)}")
    
    return combined_clean


def create_train_test_splits(df, numeric_cols, random_state=23):
    """Create train/test splits with feature engineering and standardization."""
    
    # Step 1: Data preparation
    X_raw = df[numeric_cols]
    y = df["isolate"]

    # Step 2: Sample 12k per isolate
    df_all = X_raw.copy()
    df_all["isolate"] = y

    sampled_dfs = []
    for isolate_label, group in df_all.groupby("isolate"):
        n_samples = min(len(group), 12000)
        sampled_group = group.sample(n=n_samples, random_state=random_state)
        sampled_dfs.append(sampled_group)

    df_sampled = pd.concat(sampled_dfs).reset_index(drop=True)
    X_raw = df_sampled.drop(columns="isolate")
    y = df_sampled["isolate"]

    # Step 3: Feature transformations
    X_all = pd.DataFrame(index=X_raw.index)

    # Compute offsets dynamically
    log_offsets = {}
    for col in numeric_cols:
        min_val = X_raw[col].min()
        log_offsets[col] = -min_val if min_val < 0 else 0

    for col in numeric_cols:
        x = X_raw[col]
        X_all[col] = x
        if log_offsets[col] > 0:
            X_all[f"{col}_log"] = np.log1p(x + log_offsets[col])
        X_all[f"{col}_sqrt"] = np.sqrt(np.clip(x, 0, None))

    # Step 4: Standardize all features
    scaler = StandardScaler()
    X_all_scaled = pd.DataFrame(
        scaler.fit_transform(X_all),
        columns=X_all.columns,
        index=X_all.index
    )

    # Step 5: Filter to real isolates only (remove Negatives)
    real_isolate_mask = y != "Negatives"
    X_filtered = X_all_scaled[real_isolate_mask]
    y_filtered = y[real_isolate_mask]

    # Step 6: Stratified train/test split (10k train, 2k test per isolate)
    train_dfs = []
    test_dfs = []

    for isolate_label in y_filtered.unique():
        mask = y_filtered == isolate_label
        X_iso = X_filtered[mask]
        
        # Sample up to 12k total (10k train + 2k test)
        if len(X_iso) > 12000:
            X_iso = X_iso.sample(n=12000, random_state=random_state)
        
        # Split into 10k train, 2k test
        n_train = min(10000, int(len(X_iso) * 10/12))
        n_test = min(2000, len(X_iso) - n_train)
        
        X_iso_train = X_iso.iloc[:n_train]
        X_iso_test = X_iso.iloc[n_train:n_train + n_test]
        
        train_dfs.append(X_iso_train)
        test_dfs.append(X_iso_test)

    # Combine all isolates
    X_train = pd.concat(train_dfs).reset_index(drop=True)
    X_test = pd.concat(test_dfs).reset_index(drop=True)

    # Get corresponding labels
    y_train = pd.concat([y_filtered[y_filtered == isolate].iloc[:len(train_dfs[i])] 
                          for i, isolate in enumerate(y_filtered.unique())]
                         ).reset_index(drop=True)
    y_test = pd.concat([y_filtered[y_filtered == isolate].iloc[len(train_dfs[i]):len(train_dfs[i]) + len(test_dfs[i])] 
                         for i, isolate in enumerate(y_filtered.unique())]
                        ).reset_index(drop=True)

    # Final dataframes
    df_train = X_train.copy()
    df_train["isolate"] = y_train.values

    df_test = X_test.copy()
    df_test["isolate"] = y_test.values

    print(f"Training set: {df_train.shape}")
    print(f"Test set: {df_test.shape}")
    print(f"\nTraining set isolates:\n{df_train['isolate'].value_counts()}")
    print(f"\nTest set isolates:\n{df_test['isolate'].value_counts()}")
    
    return df_train, df_test, scaler


# =========================================================
# MODEL TRAINING FUNCTIONS
# =========================================================

def select_topn_pointbiserial(X, y, n=50):
    """Select top-n features using point-biserial correlation."""
    classes = np.unique(y)
    corrs = pd.Series(index=X.columns, dtype=float)

    for feat in X.columns:
        values = X[feat].values
        max_corr = 0
        for cls in classes:
            y_bin = (y == cls).astype(int)
            if np.std(values) == 0:
                corr = 0
            else:
                corr = np.corrcoef(values, y_bin)[0,1]
                if np.isnan(corr):
                    corr = 0
            max_corr = max(max_corr, abs(corr))
        corrs[feat] = max_corr

    return corrs.sort_values(ascending=False).head(n).index.tolist()


def get_base_models():
    """Define all base models for ensemble."""
    return [
        ("RF_md4",  RandomForestClassifier(n_estimators=100, max_depth=4, class_weight="balanced", random_state=23)),
        ("RF_md8",  RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=23)),
        ("RF_md12", RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced", random_state=23)),
        ("RF_md16", RandomForestClassifier(n_estimators=400, max_depth=16, class_weight="balanced", random_state=23)),

        ("KNN_05",  KNeighborsClassifier(5)),
        ("KNN_10",  KNeighborsClassifier(10)),
        ("KNN_25",  KNeighborsClassifier(25)),
        ("KNN_50",  KNeighborsClassifier(50)),

        ("GaussianNB", GaussianNB()),

        ("MLP_50_l2",  MLPClassifier((50,),  alpha=1e-2, max_iter=500, random_state=23)),
        ("MLP_100_nopen", MLPClassifier((100,), alpha=0, max_iter=500, random_state=23)),
        ("MLP_5050_l2", MLPClassifier((50, 50), alpha=1e-2, max_iter=500, random_state=23)),
        ("MLP_100_l2hard", MLPClassifier((100,), alpha=1e-1, max_iter=500, random_state=23)),

        ("Ada50_stump", AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=50, learning_rate=0.5, random_state=23)),
        ("Ada100_depth2", AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=100, learning_rate=0.3, random_state=23)),
        ("Ada50_depth2", AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=50, learning_rate=0.5, random_state=23)),
        ("Ada50_depth3", AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=50, learning_rate=0.3, random_state=23))
    ]


def evaluate_models_on_feature_set(base_models, X_train, y_train, n_features, feature_sets):
    """Evaluate all base models on a specific feature set."""
    features = feature_sets[n_features]
    X_train_sel = X_train[features]
    
    model_results = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=23)
    
    print(f"  Evaluating base models for top {n_features} features:")
    for name, model in base_models:
        scores = []
        for tr_idx, val_idx in skf.split(X_train_sel, y_train):
            X_tr, X_val = X_train_sel.iloc[tr_idx], X_train_sel.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            m = clone(model)
            m.fit(X_tr, y_tr)
            preds = m.predict(X_val)
            scores.append(balanced_accuracy_score(y_val, preds))
        mean_score = np.mean(scores)
        print(f"    {name:18s} CV bal-acc = {mean_score:.3f}")
        model_results.append((name, model, n_features, mean_score))
    
    return model_results


def train_and_evaluate_pair(cls1, cls2, X_train, y_train, X_test, y_test, feature_sets, base_models):
    """Train and evaluate ensemble for a single pairwise comparison."""
    
    print(f"\n=== Pair: {cls1} vs {cls2} ===")

    # Subset pairwise data
    train_mask = y_train.isin([cls1, cls2])
    test_mask  = y_test.isin([cls1, cls2])

    X_train_pair = X_train[train_mask]
    y_train_pair = y_train[train_mask]

    X_test_pair = X_test[test_mask]
    y_test_pair = y_test[test_mask]

    # Evaluate models on all feature sets
    print("  Feature selection:")
    for n_features in [10, 20, 30]:
        print(f"    Top {n_features}: {len(feature_sets[n_features])} features selected")

    # Train base models on each feature set and collect results
    model_results = []
    for n_features in [10, 20, 30]:
        model_results.extend(evaluate_models_on_feature_set(base_models, X_train_pair, y_train_pair, n_features, feature_sets))

    # Select top 5 models across all feature sets
    print(f"\n  Selecting top 5 models across all feature sets:")
    model_results_sorted = sorted(model_results, key=lambda x: x[3], reverse=True)
    top5_models = model_results_sorted[:5]

    # Train top 5 models on their respective feature sets and evaluate ensemble
    ensemble_predictions = []
    ensemble_probabilities = []
    trained_models = []

    for name, model, n_features, _ in top5_models:
        features = feature_sets[n_features]
        X_train_sel = X_train_pair[features]
        X_test_sel = X_test_pair[features]

        m = clone(model)
        m.fit(X_train_sel, y_train_pair)
        preds = m.predict(X_test_sel)
        proba = m.predict_proba(X_test_sel)

        ensemble_predictions.append(preds)
        ensemble_probabilities.append(proba)
        trained_models.append((name, m, n_features, features))

    # Get class labels and create mapping
    class_labels = y_train_pair.unique()
    label_to_idx = {label: idx for idx, label in enumerate(class_labels)}
    idx_to_label = {idx: label for idx, label in enumerate(class_labels)}

    # Encode predictions to integers for voting
    encoded_predictions = []
    for preds in ensemble_predictions:
        encoded = np.array([label_to_idx[p] for p in preds])
        encoded_predictions.append(encoded)

    # Majority voting for predictions
    ensemble_pred_final = np.array(encoded_predictions).T
    y_pred_encoded = np.array([np.bincount(row).argmax() for row in ensemble_pred_final])

    # Decode back to class labels
    y_pred = np.array([idx_to_label[idx] for idx in y_pred_encoded])

    # Average probabilities
    y_proba = np.mean(ensemble_probabilities, axis=0)

    # Per-class metrics
    per_class_metrics = {}
    for i, cls in enumerate([cls1, cls2]):
        y_true_bin = (y_test_pair == cls).astype(int)
        y_pred_bin = (y_pred == cls).astype(int)
        per_class_metrics[cls] = {
            "balanced_accuracy": balanced_accuracy_score(y_true_bin, y_pred_bin),
            "f1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
            "auc": roc_auc_score(y_true_bin, y_proba[:, i])
        }

    # Aggregate pairwise metrics
    bal_vals = [m["balanced_accuracy"] for m in per_class_metrics.values()]
    f1_vals  = [m["f1"] for m in per_class_metrics.values()]
    auc_vals = [m["auc"] for m in per_class_metrics.values()]

    return {
        "pair_balanced_accuracy": np.mean(bal_vals),
        "pair_f1": np.mean(f1_vals),
        "pair_auc": np.mean(auc_vals),
        "per_class": per_class_metrics,
        "top5_models": [(name, n_features) for name, model, n_features, _ in top5_models],
        "trained_models": trained_models,
        "class_mappings": {"label_to_idx": label_to_idx, "idx_to_label": idx_to_label, "class_labels": class_labels}
    }


def save_pair_models(cls1, cls2, pair_result):
    """Save trained models and metadata for a pairwise comparison."""
    pair_dir = f"models/{cls1}_vs_{cls2}"
    os.makedirs(pair_dir, exist_ok=True)

    # Save the 5 base models with their feature sets
    for name, model, n_features, features in pair_result["trained_models"]:
        model_path = f"{pair_dir}/{name}_top{n_features}.pkl"
        joblib.dump(model, model_path)
        
        features_path = f"{pair_dir}/{name}_top{n_features}_features.pkl"
        joblib.dump(features, features_path)

    # Save class label mappings
    joblib.dump(pair_result["class_mappings"], f"{pair_dir}/class_mappings.pkl")

    # Save ensemble metadata (which models are used)
    ensemble_metadata = {
        "models": [(name, n_features) for name, _, n_features, _ in pair_result["trained_models"]],
        "pair": (cls1, cls2)
    }
    joblib.dump(ensemble_metadata, f"{pair_dir}/ensemble_metadata.pkl")

    print(f"  ✓ Models saved to {pair_dir}/")


# =========================================================
# MAIN FUNCTION
# =========================================================
def main():
    """Main execution function."""
    
    print("=" * 70)
    print("PAIRWISE ENSEMBLE CLASSIFIER TRAINING")
    print("=" * 70)
    
    # -------------------------
    # Data Loading and Preparation
    # -------------------------
    print("\n1. Loading data...")
    combined_df = load_and_combine_data()
    print(combined_df.shape)
    print(combined_df.columns)
    print(combined_df.groupby("isolate").size())
    
    print("\n2. Removing low-count isolates...")
    remove_list = []
    combined_df = remove_low_count_isolates(combined_df, remove_list)
    
    print("\n3. Cleaning with Isolation Forest...")
    numeric_cols = combined_df.select_dtypes(include=["number"]).columns.tolist()
    combined_df = clean_with_isolation_forest(combined_df, numeric_cols)
    
    print("\n4. Creating train/test splits with feature engineering...")
    df_train, df_test, scaler = create_train_test_splits(combined_df, numeric_cols)
    
    # -------------------------
    # Prepare data for modeling
    # -------------------------
    X_train = df_train.drop(columns="isolate")
    y_train = df_train["isolate"]

    X_test = df_test.drop(columns="isolate")
    y_test = df_test["isolate"]
    
    # -------------------------
    # Get base models
    # -------------------------
    base_models = get_base_models()
    
    # -------------------------
    # Create models directory
    # -------------------------
    os.makedirs("models", exist_ok=True)
    
    # -------------------------
    # Main pairwise sweep
    # -------------------------
    print("\n5. Training pairwise ensembles...")
    isolate_pairs = list(combinations(np.unique(y_train), 2))
    pairwise_results = {}
    all_stats = []

    for pair_idx, (cls1, cls2) in enumerate(isolate_pairs):
        print(f"\n[{pair_idx + 1}/{len(isolate_pairs)}]", end="")
        
        # -------------------------
        # Prepare feature sets for this pair
        # -------------------------
        print(f"\n=== Pair: {cls1} vs {cls2} ===")
        
        # Subset pairwise data
        train_mask = y_train.isin([cls1, cls2])
        X_train_pair = X_train[train_mask]
        y_train_pair = y_train[train_mask]
        
        print("  Feature selection:")
        feature_sets = {}
        for n_features in [10, 20, 30]:
            top_features = select_topn_pointbiserial(X_train_pair, y_train_pair, n=n_features)
            feature_sets[n_features] = top_features
            print(f"    Top {n_features}: {len(top_features)} features selected")
        
        # Train and evaluate pair
        pair_result = train_and_evaluate_pair(
            cls1, cls2, X_train, y_train, X_test, y_test, feature_sets, base_models
        )
        
        pairwise_results[(cls1, cls2)] = pair_result
        
        # Save models
        save_pair_models(cls1, cls2, pair_result)
        
        # Add to stats list (one row per pair)
        all_stats.append({
            "class_1": cls1,
            "class_2": cls2,
            "balanced_accuracy": pair_result["pair_balanced_accuracy"],
            "f1_score": pair_result["pair_f1"],
            "auc": pair_result["pair_auc"]
        })
    
    # -------------------------
    # Save all pairwise statistics to CSV
    # -------------------------
    print("\n6. Saving statistics...")
    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv("all_pairwise_stats_round01.csv", index=False)
    print(f"✓ Statistics saved to all_pairwise_stats.csv")
    
    # -------------------------
    # Print summary
    # -------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for (cls1, cls2), res in pairwise_results.items():
        print(f"\nPair {cls1} vs {cls2}")
        print(f"  Pair balanced accuracy: {res['pair_balanced_accuracy']:.3f}")
        print(f"  Pair F1:               {res['pair_f1']:.3f}")
        print(f"  Pair AUC:              {res['pair_auc']:.3f}")
        print("  Per-class metrics:")
        for cls, m in res["per_class"].items():
            print(
                f"    {cls}: BalancedAcc={m['balanced_accuracy']:.3f}, "
                f"F1={m['f1']:.3f}, AUC={m['auc']:.3f}"
            )
        print("  Top 5 models in ensemble:")
        for name, n_features in res["top5_models"]:
            print(f"    {name} (top {n_features} features)")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
