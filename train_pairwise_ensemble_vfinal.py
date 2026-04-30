import argparse
import os
import re
import numpy as np
import pandas as pd
from itertools import combinations

import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

FILENAME_RE = re.compile(r"(round\d+)_(\d+C)_(d\d+)_([0-9.]+)_([A-H][0-9]+)_([^\.]+)\.csv")

def normalize_round_str(x: str) -> str:
    s = str(x).strip().lower()
    m = re.match(r"^round0*([0-9]+)$", s)
    return f"round{int(m.group(1))}" if m else s

def load_and_combine_data(r: int):
    """Load monoculture CSVs for the given round number (1, 2, …)."""
    nn = f"{r:02d}"
    data_dir = f"../data/round{nn}"
    metadata_path = os.path.join(data_dir, "all_metadata.csv")
    pairs_to_wells_path = os.path.join(data_dir, "pairs_to_wells.csv")
    # non-zero-padded round string used in filenames, e.g. "round1"
    round_prefix = f"round{r}"

    metadata = pd.read_csv(metadata_path).drop_duplicates()
    pairs_to_wells = pd.read_csv(pairs_to_wells_path).drop_duplicates()

    pairs_to_wells["Round"] = pairs_to_wells["Round"].apply(normalize_round_str)
    pairs_to_wells["Day"] = pairs_to_wells["Day"].astype(str).str.zfill(2)
    pairs_to_wells["Day"] = "d" + pairs_to_wells["Day"].str[-2:]
    pairs_to_wells["Well"] = pairs_to_wells["Well"].astype(str).str.strip()

    mono_meta = metadata[metadata["Type"] == "mono"]
    community_to_isolate = dict(zip(mono_meta["Community"], mono_meta["IsolateA"]))

    csv_records = []
    for folder in os.listdir(data_dir):
        if not folder.endswith("2026"):
            continue
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(".csv"):
                continue
            # filter to files belonging to this round (non-zero-padded prefix)
            if not fname.startswith(round_prefix + "_"):
                continue
            m = FILENAME_RE.match(fname)
            if not m:
                continue
            round_n, temp, day, date, well, sample = m.groups()
            csv_records.append({
                "filepath": os.path.join(folder_path, fname),
                "filename": fname,
                "Round": normalize_round_str(round_n),
                "Temp": temp,
                "Day": day,
                "Date": date,
                "Well": well.strip(),
                "Sample": sample,
                "Folder": folder,
            })

    csv_index = pd.DataFrame(csv_records)
    csv_index = csv_index.merge(
        pairs_to_wells[["Round", "Day", "Well", "Community"]],
        on=["Round", "Day", "Well"],
        how="left",
        validate="many_to_one",
    )

    # keep only monoculture files
    csv_index = csv_index[csv_index["Community"].isin(set(community_to_isolate.keys()))].copy()

    dfs = []
    for _, rec in csv_index.iterrows():
        df = pd.read_csv(rec["filepath"])
        df["Temp"] = rec["Temp"]
        df["Day"] = rec["Day"]
        df["Date"] = rec["Date"]
        df["Well"] = rec["Well"]
        df["filename"] = rec["filename"]
        df["isolate"] = community_to_isolate[rec["Community"]]
        df["sample"] = rec["Sample"]
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No monoculture data found after mapping pairs_to_wells + metadata.")

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = pd.get_dummies(combined_df, columns=["Temp", "Day"])
    return combined_df

def remove_low_count_isolates(df, remove_list):
    df_filtered = df[~df["isolate"].isin(remove_list)]
    print(f"Removed isolates: {remove_list}")
    print(f"Shape after removal: {df_filtered.shape}")
    return df_filtered

def clean_with_isolation_forest(df, numeric_cols, contamination=0.05, random_state=42):
    from sklearn.ensemble import IsolationForest
    cleaned = []
    for iso, g in df.groupby("isolate"):
        iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
        iso_forest.fit(g[numeric_cols])
        inliers = iso_forest.predict(g[numeric_cols]) == 1
        cleaned.append(g[inliers])
    out = pd.concat(cleaned).reset_index(drop=True).dropna()
    print(f"Original rows: {len(df)}, Cleaned rows: {len(out)}")
    return out

def feature_engineer(df, numeric_cols, log_offsets):
    X = df[numeric_cols].copy()
    for col in numeric_cols:
        off = log_offsets[col]
        X[f"{col}_log"] = np.log1p(np.clip(X[col] + off, 0, None))
        X[f"{col}_sqrt"] = np.sqrt(np.clip(X[col], 0, None))
    return X

def create_train_test_splits(df, numeric_cols, random_state=23, n_train=20000, n_test=5000):
    df_filtered = df[df["isolate"] != "Negatives"].copy()

    train_list, test_list = [], []
    for iso, g in df_filtered.groupby("isolate"):
        g = g.sample(frac=1, random_state=random_state).reset_index(drop=True)
        if len(g) < (n_train + n_test):
            raise ValueError(f"Not enough data for isolate {iso}: found {len(g)}, require {n_train + n_test}")
        train_list.append(g.iloc[:n_train])
        test_list.append(g.iloc[n_train:n_train + n_test])

    df_train = pd.concat(train_list, ignore_index=True)
    df_test = pd.concat(test_list, ignore_index=True)

    train_log_offsets = {col: (-df_train[col].min() if df_train[col].min() < 0 else 0) for col in numeric_cols}

    X_train_feat = feature_engineer(df_train, numeric_cols, train_log_offsets)
    X_test_feat = feature_engineer(df_test, numeric_cols, train_log_offsets)

    engineered_cols = list(X_train_feat.columns)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_feat), columns=engineered_cols, index=X_train_feat.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_feat), columns=engineered_cols, index=X_test_feat.index)

    df_train_final = X_train_scaled.copy()
    df_train_final["isolate"] = df_train["isolate"].values

    df_test_final = X_test_scaled.copy()
    df_test_final["isolate"] = df_test["isolate"].values

    return df_train_final, df_test_final, scaler, train_log_offsets, engineered_cols

def select_topn_pointbiserial(X, y, n=50):
    classes = np.unique(y)
    corrs = pd.Series(index=X.columns, dtype=float)
    for feat in X.columns:
        values = X[feat].values
        max_corr = 0.0
        for cls in classes:
            y_bin = (y == cls).astype(int)
            if np.std(values) == 0:
                corr = 0.0
            else:
                corr = np.corrcoef(values, y_bin)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
            max_corr = max(max_corr, abs(corr))
        corrs[feat] = max_corr
    return corrs.sort_values(ascending=False).head(n).index.tolist()

def get_base_models():
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

        ("MLP_50_l2",  MLPClassifier((50,), alpha=1e-2, max_iter=500, random_state=23)),
        ("MLP_100_nopen", MLPClassifier((100,), alpha=0, max_iter=500, random_state=23)),
        ("MLP_5050_l2", MLPClassifier((50, 50), alpha=1e-2, max_iter=500, random_state=23)),
        ("MLP_100_l2hard", MLPClassifier((100,), alpha=1e-1, max_iter=500, random_state=23)),

        ("Ada50_stump", AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=50, learning_rate=0.5, random_state=23)),
        ("Ada100_depth2", AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=100, learning_rate=0.3, random_state=23)),
        ("Ada50_depth2", AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=50, learning_rate=0.5, random_state=23)),
        ("Ada50_depth3", AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=50, learning_rate=0.3, random_state=23)),
    ]

def evaluate_models_on_feature_set(base_models, X_train, y_train, n_features, feature_sets):
    feats = feature_sets[n_features]
    X_sel = X_train[feats]
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=23)

    results = []
    for name, model in base_models:
        scores = []
        for tr_idx, val_idx in skf.split(X_sel, y_train):
            m = clone(model)
            m.fit(X_sel.iloc[tr_idx], y_train.iloc[tr_idx])
            preds = m.predict(X_sel.iloc[val_idx])
            scores.append(balanced_accuracy_score(y_train.iloc[val_idx], preds))
        results.append((name, model, n_features, float(np.mean(scores))))
    return results

def train_and_evaluate_pair(cls1, cls2, X_train, y_train, X_test, y_test, feature_sets, base_models):
    train_mask = y_train.isin([cls1, cls2])
    test_mask = y_test.isin([cls1, cls2])

    X_train_pair = X_train[train_mask]
    y_train_pair = y_train[train_mask]
    X_test_pair = X_test[test_mask]
    y_test_pair = y_test[test_mask]

    model_results = []
    for n in [10, 20, 30]:
        model_results.extend(evaluate_models_on_feature_set(base_models, X_train_pair, y_train_pair, n, feature_sets))

    top5 = sorted(model_results, key=lambda x: x[3], reverse=True)[:5]

    preds_list, probas_list, trained_models = [], [], []
    for name, model, n_features, _ in top5:
        feats = feature_sets[n_features]
        m = clone(model)
        m.fit(X_train_pair[feats], y_train_pair)
        preds_list.append(m.predict(X_test_pair[feats]))
        probas_list.append(m.predict_proba(X_test_pair[feats]))
        trained_models.append((name, m, n_features, feats))

    class_labels = y_train_pair.unique()
    label_to_idx = {lab: i for i, lab in enumerate(class_labels)}
    idx_to_label = {i: lab for i, lab in enumerate(class_labels)}

    enc = np.vstack([[label_to_idx[p] for p in preds] for preds in preds_list]).T
    y_pred_encoded = np.array([np.bincount(row).argmax() for row in enc])
    y_pred = np.array([idx_to_label[i] for i in y_pred_encoded])

    y_proba = np.mean(probas_list, axis=0)

    per_class = {}
    for i, cls in enumerate([cls1, cls2]):
        y_true_bin = (y_test_pair == cls).astype(int)
        y_pred_bin = (y_pred == cls).astype(int)
        per_class[cls] = {
            "balanced_accuracy": balanced_accuracy_score(y_true_bin, y_pred_bin),
            "f1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
            "auc": roc_auc_score(y_true_bin, y_proba[:, i]),
        }

    return {
        "pair_balanced_accuracy": float(np.mean([m["balanced_accuracy"] for m in per_class.values()])),
        "pair_f1": float(np.mean([m["f1"] for m in per_class.values()])),
        "pair_auc": float(np.mean([m["auc"] for m in per_class.values()])),
        "per_class": per_class,
        "top5_models": [(name, n_features) for name, _, n_features, _ in top5],
        "trained_models": trained_models,
        "class_mappings": {"label_to_idx": label_to_idx, "idx_to_label": idx_to_label, "class_labels": class_labels},
    }

def save_pair_models(cls1, cls2, pair_result, models_root):
    pair_dir = os.path.join(models_root, f"{cls1}_vs_{cls2}")
    os.makedirs(pair_dir, exist_ok=True)

    for name, model, n_features, feats in pair_result["trained_models"]:
        joblib.dump(model, os.path.join(pair_dir, f"{name}_top{n_features}.pkl"))
        joblib.dump(feats, os.path.join(pair_dir, f"{name}_top{n_features}_features.pkl"))

    joblib.dump(pair_result["class_mappings"], os.path.join(pair_dir, "class_mappings.pkl"))
    joblib.dump(
        {"models": [(name, n_features) for name, _, n_features, _ in pair_result["trained_models"]],
         "pair": (cls1, cls2)},
        os.path.join(pair_dir, "ensemble_metadata.pkl"),
    )

def main():
    parser = argparse.ArgumentParser(description="Train pairwise ensemble classifiers for flow cytometry data.")
    parser.add_argument("-r", "--round", type=int, required=True,
                        help="Experimental round number (e.g. 1, 2). Paths use zero-padded form (round01, round02, ...).")
    args = parser.parse_args()

    r = args.round
    nn = f"{r:02d}"
    models_root = f"round{nn}_models"
    os.makedirs(models_root, exist_ok=True)

    print("=" * 70)
    print(f"PAIRWISE ENSEMBLE CLASSIFIER TRAINING  —  round {nn}")
    print("=" * 70)

    print("\n1. Loading data...")
    combined_df = load_and_combine_data(r)
    combined_df = remove_low_count_isolates(combined_df, remove_list=[])

    print("\n2. Cleaning with Isolation Forest...")
    numeric_cols = combined_df.select_dtypes(include=["number"]).columns.tolist()
    combined_df = clean_with_isolation_forest(combined_df, numeric_cols)

    print("\n3. Creating train/test splits and engineering features...")
    df_train, df_test, scaler, log_offsets, engineered_cols = create_train_test_splits(combined_df, numeric_cols)

    # Save inference-critical artifacts (no round prefix — they live under round{NN}_models/)
    joblib.dump(scaler,          os.path.join(models_root, "scaler.pkl"))
    joblib.dump(log_offsets,     os.path.join(models_root, "log_offsets.pkl"))
    joblib.dump(numeric_cols,    os.path.join(models_root, "numeric_cols.pkl"))
    joblib.dump(engineered_cols, os.path.join(models_root, "engineered_feature_columns.pkl"))
    print(f"✓ Saved inference artifacts to {models_root}/")

    X_train, y_train = df_train.drop(columns="isolate"), df_train["isolate"]
    X_test, y_test = df_test.drop(columns="isolate"), df_test["isolate"]

    base_models = get_base_models()
    isolate_pairs = list(combinations(np.unique(y_train), 2))

    print(f"\n4. Training {len(isolate_pairs)} pairwise models...")
    all_stats = []
    for cls1, cls2 in isolate_pairs:
        mask = y_train.isin([cls1, cls2])
        X_pair, y_pair = X_train[mask], y_train[mask]

        feature_sets = {n: select_topn_pointbiserial(X_pair, y_pair, n=n) for n in [10, 20, 30]}
        pair_result = train_and_evaluate_pair(cls1, cls2, X_train, y_train, X_test, y_test, feature_sets, base_models)

        save_pair_models(cls1, cls2, pair_result, models_root=models_root)

        pc = pair_result["per_class"]
        all_stats.append({
            "class_1": cls1,
            "class_2": cls2,
            "pair_balanced_accuracy": pair_result["pair_balanced_accuracy"],
            "pair_f1": pair_result["pair_f1"],
            "pair_auc": pair_result["pair_auc"],
            "balanced_accuracy_Isolate_A": pc[cls1]["balanced_accuracy"],
            "balanced_accuracy_Isolate_B": pc[cls2]["balanced_accuracy"],
            "AUC_Isolate_A": pc[cls1]["auc"],
            "AUC_Isolate_B": pc[cls2]["auc"],
            "F1_Isolate_A": pc[cls1]["f1"],
            "F1_Isolate_B": pc[cls2]["f1"],
        })
        print(f"  ✓ {cls1} vs {cls2}  ba={pair_result['pair_balanced_accuracy']:.3f}")

    stats_csv = f"all_pairwise_stats_round{nn}.csv"
    pd.DataFrame(all_stats).to_csv(stats_csv, index=False)
    print(f"\n5. Saved statistics to {stats_csv}")
    print(f"✓ Training complete. Models + inference artifacts → {models_root}/")

if __name__ == "__main__":
    main()
