import os
import re
import argparse
import numpy as np
import pandas as pd
from itertools import combinations

import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Expected filename format: classifier_test_E5_Sample53.csv
FILENAME_RE = re.compile(r"^classifier_test_([A-H][0-9]+)_([^\.]+)\.csv$", re.IGNORECASE)

def class_probability_totals(probs_by_model: list, class_names: list) -> dict:
    """Aggregate probabilities across models and rows."""
    mat = np.stack(probs_by_model)         # (n_models, n_rows, n_classes)
    mean_probs = np.mean(mat, axis=0)      # (n_rows, n_classes)
    totals = np.sum(mean_probs, axis=0)    # (n_classes,)
    return dict(zip(class_names, totals))

def build_csv_index(data_dir: str) -> pd.DataFrame:
    rows = []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if not fname.lower().endswith(".csv"):
                continue
            m = FILENAME_RE.match(fname)
            if not m:
                continue
            well, sample = m.groups()
            rows.append({
                "filepath": os.path.join(root, fname),
                "filename": fname,
                "Well": str(well).strip(),
                "Sample": sample,
            })
    return pd.DataFrame(rows)


def load_metadata_and_pairs(data_dir: str):
    metadata = pd.read_csv(os.path.join(data_dir, "all_metadata.csv")).drop_duplicates()
    pairs_to_wells = pd.read_csv(os.path.join(data_dir, "pairs_to_wells.csv")).drop_duplicates()

    if "Well" not in pairs_to_wells.columns:
        raise ValueError("pairs_to_wells.csv must contain column: Well")
    if "Community" not in pairs_to_wells.columns:
        raise ValueError("pairs_to_wells.csv must contain column: Community")

    pairs_to_wells["Well"] = pairs_to_wells["Well"].astype(str).str.strip()
    return metadata, pairs_to_wells


def merge_csv_with_pairs(csv_index: pd.DataFrame, pairs_to_wells: pd.DataFrame) -> pd.DataFrame:
    return csv_index.merge(
        pairs_to_wells[["Well", "Community"]].drop_duplicates(),
        on="Well",
        how="left",
        validate="many_to_one",
    )


def load_combined_monoculture(data_dir: str) -> pd.DataFrame:
    metadata, pairs_to_wells = load_metadata_and_pairs(data_dir)

    mono_meta = metadata[metadata["Type"] == "mono"].copy()
    if mono_meta.empty:
        raise RuntimeError("No mono rows found in all_metadata.csv (Type == 'mono').")

    if "Community" not in mono_meta.columns or "IsolateA" not in mono_meta.columns:
        raise ValueError("all_metadata.csv mono rows must include Community and IsolateA.")

    community_to_isolate = dict(zip(mono_meta["Community"], mono_meta["IsolateA"]))

    csv_index = build_csv_index(data_dir)
    if csv_index.empty:
        raise RuntimeError("No classifier_test_* CSV files found.")

    csv_index = merge_csv_with_pairs(csv_index, pairs_to_wells)

    mono_csvs = csv_index[csv_index["Community"].isin(set(community_to_isolate.keys()))].copy()
    if mono_csvs.empty:
        raise RuntimeError("No monoculture CSVs found after Well->Community mapping.")

    dfs = []
    for _, rec in mono_csvs.iterrows():
        df = pd.read_csv(rec["filepath"])
        df["isolate"] = community_to_isolate[rec["Community"]]
        df["Well"] = rec["Well"]
        df["Sample"] = rec["Sample"]
        df["filename"] = rec["filename"]
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna()
    return combined


def clean_with_isolation_forest(df, numeric_cols, contamination=0.05, random_state=42):
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
            raise ValueError(f"Not enough data for isolate {iso}: found {len(g)}, require {n_train+n_test}")
        train_list.append(g.iloc[:n_train])
        test_list.append(g.iloc[n_train:n_train + n_test])

    df_train = pd.concat(train_list, ignore_index=True)
    df_test = pd.concat(test_list, ignore_index=True)

    train_log_offsets = {
        col: (-df_train[col].min() if df_train[col].min() < 0 else 0)
        for col in numeric_cols
    }

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

        ("MLP_16_l2",      MLPClassifier((16,), alpha=1e-2, max_iter=400, random_state=23)),
        ("MLP_32_nopen",   MLPClassifier((32,), alpha=0, max_iter=400, random_state=23)),
        ("MLP_1616_l2",    MLPClassifier((16, 16), alpha=1e-2, max_iter=400, random_state=23)),
        ("MLP_64_l2hard",  MLPClassifier((64,), alpha=1e-1, max_iter=400, random_state=23)),

        ("Ada50_stump",    AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=50, learning_rate=0.5, random_state=23)),
        ("Ada100_depth2",  AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=100, learning_rate=0.3, random_state=23)),
        ("Ada50_depth2",   AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=50, learning_rate=0.5, random_state=23)),
        ("Ada50_depth3",   AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=50, learning_rate=0.3, random_state=23)),
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
        "trained_models": trained_models,
    }


def save_pair_models(cls1, cls2, pair_result, models_root):
    pair_dir = os.path.join(models_root, f"{cls1}_vs_{cls2}")
    os.makedirs(pair_dir, exist_ok=True)

    model_specs = []
    for name, model, n_features, feats in pair_result["trained_models"]:
        joblib.dump(model, os.path.join(pair_dir, f"{name}_top{n_features}.pkl"))
        joblib.dump(feats, os.path.join(pair_dir, f"{name}_top{n_features}_features.pkl"))
        model_specs.append((name, n_features))

    joblib.dump({"models": model_specs, "pair": (cls1, cls2)},
                os.path.join(pair_dir, "ensemble_metadata.pkl"))


def resolve_pair_dir(models_root: str, isolate_A: str, isolate_B: str):
    d1 = os.path.join(models_root, f"{isolate_A}_vs_{isolate_B}")
    d2 = os.path.join(models_root, f"{isolate_B}_vs_{isolate_A}")
    if os.path.isdir(d1):
        return d1
    if os.path.isdir(d2):
        return d2
    return None


def majority_vote(preds_by_model):
    mat = np.vstack(preds_by_model)
    out = []
    for j in range(mat.shape[1]):
        vals, counts = np.unique(mat[:, j], return_counts=True)
        out.append(vals[np.argmax(counts)])
    return np.array(out)


def apply_pair_model_to_well(X_scaled, isolate_A, isolate_B, models_root):
    pair_dir = resolve_pair_dir(models_root, isolate_A, isolate_B)
    if pair_dir is None:
        raise FileNotFoundError(f"No model directory found for {isolate_A} vs {isolate_B}")

    meta = joblib.load(os.path.join(pair_dir, "ensemble_metadata.pkl"))
    model_specs = meta["models"]

    preds_by_model = []
    probas_by_model = []
    first_classes = None

    for model_name, n_features in model_specs:
        model = joblib.load(os.path.join(pair_dir, f"{model_name}_top{n_features}.pkl"))
        feats = joblib.load(os.path.join(pair_dir, f"{model_name}_top{n_features}_features.pkl"))
        missing = [c for c in feats if c not in X_scaled.columns]
        if missing:
            raise KeyError(f"Missing feature columns: {missing[:10]}")

        X_sel = X_scaled[feats]
        preds_by_model.append(model.predict(X_sel))

        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_sel)
            probas_by_model.append(probas)
            if first_classes is None:
                first_classes = list(getattr(model, "classes_"))

    y_pred = majority_vote(preds_by_model)
    count_A = int(np.sum(y_pred == isolate_A))
    count_B = int(np.sum(y_pred == isolate_B))
    N = int(len(y_pred))

    summed_proba_A = np.nan
    summed_proba_B = np.nan
    if probas_by_model and first_classes is not None:
        totals = class_probability_totals(probas_by_model, first_classes)
        summed_proba_A = float(totals.get(isolate_A, np.nan))
        summed_proba_B = float(totals.get(isolate_B, np.nan))

    return count_A, count_B, N, summed_proba_A, summed_proba_B

def run_training(data_dir: str, models_root: str, n_train: int, n_test: int):
    os.makedirs(models_root, exist_ok=True)

    mono_df = load_combined_monoculture(data_dir)
    numeric_cols = mono_df.select_dtypes(include=["number"]).columns.tolist()
    mono_df = clean_with_isolation_forest(mono_df, numeric_cols)

    df_train, df_test, scaler, log_offsets, engineered_cols = create_train_test_splits(
        mono_df, numeric_cols, n_train=n_train, n_test=n_test
    )

    joblib.dump(scaler, os.path.join(models_root, "scaler.pkl"))
    joblib.dump(log_offsets, os.path.join(models_root, "log_offsets.pkl"))
    joblib.dump(numeric_cols, os.path.join(models_root, "numeric_cols.pkl"))
    joblib.dump(engineered_cols, os.path.join(models_root, "engineered_feature_columns.pkl"))

    X_train, y_train = df_train.drop(columns="isolate"), df_train["isolate"]
    X_test, y_test = df_test.drop(columns="isolate"), df_test["isolate"]

    base_models = get_base_models()
    isolate_pairs = list(combinations(np.unique(y_train), 2))

    all_stats = []
    for cls1, cls2 in isolate_pairs:
        mask = y_train.isin([cls1, cls2])
        X_pair, y_pair = X_train[mask], y_train[mask]

        feature_sets = {n: select_topn_pointbiserial(X_pair, y_pair, n=n) for n in [10, 20, 30]}
        pair_result = train_and_evaluate_pair(cls1, cls2, X_train, y_train, X_test, y_test, feature_sets, base_models)

        save_pair_models(cls1, cls2, pair_result, models_root=models_root)

        all_stats.append({
            "class_1": cls1,
            "class_2": cls2,
            "balanced_accuracy": pair_result["pair_balanced_accuracy"],
            "f1_score": pair_result["pair_f1"],
            "auc": pair_result["pair_auc"],
        })

    stats_path = os.path.join(models_root, "all_pairwise_stats.csv")
    pd.DataFrame(all_stats).to_csv(stats_path, index=False)
    print(f"Saved training stats: {stats_path}")


def run_inference(data_dir: str, models_root: str, output_csv: str):
    metadata, pairs_to_wells = load_metadata_and_pairs(data_dir)

    scaler = joblib.load(os.path.join(models_root, "scaler.pkl"))
    log_offsets = joblib.load(os.path.join(models_root, "log_offsets.pkl"))
    numeric_cols = joblib.load(os.path.join(models_root, "numeric_cols.pkl"))
    engineered_cols = joblib.load(os.path.join(models_root, "engineered_feature_columns.pkl"))

    pair_meta = metadata[metadata["Type"] == "pair"].copy()
    if pair_meta.empty:
        raise RuntimeError("No pair rows found in all_metadata.csv (Type == 'pair').")
    if not {"Community", "IsolateA", "IsolateB"}.issubset(pair_meta.columns):
        raise ValueError("Pair metadata must include Community, IsolateA, IsolateB.")

    csv_index = build_csv_index(data_dir)
    csv_index = merge_csv_with_pairs(csv_index, pairs_to_wells)

    pair_wells = csv_index.merge(
        pair_meta[["Community", "IsolateA", "IsolateB"]].drop_duplicates(),
        on="Community",
        how="inner",
        validate="many_to_one",
    )

    out = []
    for _, rec in pair_wells.iterrows():
        df_raw = pd.read_csv(rec["filepath"])

        X = df_raw[numeric_cols].copy()
        for col in numeric_cols:
            off = log_offsets[col]
            X[f"{col}_log"] = np.log1p(np.clip(X[col] + off, 0, None))
            X[f"{col}_sqrt"] = np.sqrt(np.clip(X[col], 0, None))

        X = X[engineered_cols]
        X_scaled = pd.DataFrame(scaler.transform(X), columns=engineered_cols, index=X.index)

        count_A, count_B, N, summed_proba_A, summed_proba_B = apply_pair_model_to_well(
                     X_scaled=X_scaled,
                     isolate_A=rec["IsolateA"],
                     isolate_B=rec["IsolateB"],
                     models_root=models_root,
                     )

        out.append({
            "Well": rec["Well"],
            "Sample": rec["Sample"],
            "Community": rec["Community"],
            "isolate_A": rec["IsolateA"],
            "isolate_B": rec["IsolateB"],
            "count_A": count_A,
            "count_B": count_B,
            "summed_proba_A": summed_proba_A,
            "summed_proba_B": summed_proba_B,
            "N_events": N,
            "filename": rec["filename"],
        })

    df_out = pd.DataFrame(out)
    df_out.to_csv(output_csv, index=False)
    print(f"Saved {output_csv} (rows={len(df_out)})")
    if not df_out.empty:
        print(df_out.head())


def main():
    parser = argparse.ArgumentParser(description="Train + apply pairwise ensemble for classifier_test_* CSVs")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing all_metadata.csv, pairs_to_wells.csv, and classifier_test_*.csv files")
    parser.add_argument("--models-root", type=str, default="validate_models_artifacts",
                        help="Directory to save/load model artifacts")
    parser.add_argument("--output-csv", type=str, default="coculture_pairwise_counts_validate.csv",
                        help="Output CSV path for co-culture predictions")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training and only run inference using existing artifacts")
    parser.add_argument("--n-train", type=int, default=20000,
                        help="Train events per isolate")
    parser.add_argument("--n-test", type=int, default=5000,
                        help="Test events per isolate")
    args = parser.parse_args()

    if not args.skip_train:
        run_training(
            data_dir=args.data_dir,
            models_root=args.models_root,
            n_train=args.n_train,
            n_test=args.n_test,
        )

    run_inference(
        data_dir=args.data_dir,
        models_root=args.models_root,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
