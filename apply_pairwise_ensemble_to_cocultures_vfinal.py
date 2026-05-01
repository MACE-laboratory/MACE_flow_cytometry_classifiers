import argparse
import os
import re
import numpy as np
import pandas as pd
import joblib

FILENAME_RE = re.compile(r"(round\d+)_(\d+C)_(d\d+)_([0-9.]+)_([A-H][0-9]+)_([^\.]+)\.csv")


def normalize_round_str(x: str) -> str:
    s = str(x).strip().lower()
    m = re.match(r"^round0*([0-9]+)$", s)
    return f"round{int(m.group(1))}" if m else s


def build_csv_index(data_dir: str, round_prefix: str) -> pd.DataFrame:
    """Index per-well CSVs in data_dir/*2026/ that match the given round prefix."""
    rows = []
    for folder in os.listdir(data_dir):
        if not folder.endswith("2026"):
            continue
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(".csv"):
                continue
            if not fname.startswith(round_prefix + "_"):
                continue
            m = FILENAME_RE.match(fname)
            if not m:
                continue
            round_n, temp, day, date, well, sample = m.groups()
            rows.append({
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
    return pd.DataFrame(rows)


def load_pair_wells_index(r: int) -> pd.DataFrame:
    """Return a dataframe of co-culture well CSV records with IsolateA / IsolateB attached."""
    nn = f"{r:02d}"
    data_dir = f"../data/round{nn}"
    round_prefix = f"round{r}"

    metadata = pd.read_csv(os.path.join(data_dir, "all_metadata.csv")).drop_duplicates()
    pair_meta = metadata[metadata["Type"] == "pair"].copy()

    pairs_to_wells = pd.read_csv(os.path.join(data_dir, "pairs_to_wells.csv")).drop_duplicates()
    pairs_to_wells["Round"] = pairs_to_wells["Round"].apply(normalize_round_str)
    pairs_to_wells["Day"] = pairs_to_wells["Day"].astype(str).str.zfill(2)
    pairs_to_wells["Day"] = "d" + pairs_to_wells["Day"].str[-2:]
    pairs_to_wells["Well"] = pairs_to_wells["Well"].astype(str).str.strip()

    csv_index = build_csv_index(data_dir, round_prefix)

    csv_index = csv_index.merge(
        pairs_to_wells[["Round", "Day", "Well", "Community"]],
        on=["Round", "Day", "Well"],
        how="left",
        validate="many_to_one",
    )

    pair_wells = csv_index.merge(
        pair_meta[["Community", "IsolateA", "IsolateB"]],
        on="Community",
        how="inner",
        validate="many_to_one",
    )

    return pair_wells


def feature_engineer_raw(df_raw: pd.DataFrame, numeric_cols: list, log_offsets: dict) -> pd.DataFrame:
    X = df_raw[numeric_cols].copy()
    for col in numeric_cols:
        off = log_offsets[col]
        X[f"{col}_log"] = np.log1p(np.clip(X[col] + off, 0, None))
        X[f"{col}_sqrt"] = np.sqrt(np.clip(X[col], 0, None))
    return X


def majority_vote(preds_by_model: list) -> np.ndarray:
    mat = np.vstack(preds_by_model)  # (n_models, n_rows)
    out = []
    for j in range(mat.shape[1]):
        vals, counts = np.unique(mat[:, j], return_counts=True)
        out.append(vals[np.argmax(counts)])
    return np.array(out)


def class_probability_totals(probs_by_model: list, class_names: list) -> dict:
    """Aggregate probabilities across models and rows.

    probs_by_model: list of (n_rows, n_classes)
    class_names: list of class labels (length = n_classes)

    Returns:
        dict: {class_name: sum of averaged probabilities across rows}
    """
    mat = np.stack(probs_by_model)         # (n_models, n_rows, n_classes)
    mean_probs = np.mean(mat, axis=0)      # (n_rows, n_classes)
    totals = np.sum(mean_probs, axis=0)    # (n_classes,)
    return dict(zip(class_names, totals))


def resolve_pair_dir(models_root: str, isolate_A: str, isolate_B: str):
    d1 = os.path.join(models_root, f"{isolate_A}_vs_{isolate_B}")
    d2 = os.path.join(models_root, f"{isolate_B}_vs_{isolate_A}")
    if os.path.isdir(d1):
        return d1
    if os.path.isdir(d2):
        return d2
    return None


def apply_pair_model_to_well(X_scaled: pd.DataFrame, isolate_A: str, isolate_B: str, models_root: str) -> dict:
    pair_dir = resolve_pair_dir(models_root, isolate_A, isolate_B)
    if pair_dir is None:
        raise FileNotFoundError(
            f"No model directory found for {isolate_A} vs {isolate_B} under {models_root}/"
        )

    meta = joblib.load(os.path.join(pair_dir, "ensemble_metadata.pkl"))
    model_specs = meta["models"]  # list of (name, n_features)

    preds_by_model = []
    probas_by_model = []

    for model_name, n_features in model_specs:
        model_path = os.path.join(pair_dir, f"{model_name}_top{n_features}.pkl")
        feat_path = os.path.join(pair_dir, f"{model_name}_top{n_features}_features.pkl")

        model = joblib.load(model_path)
        feats = joblib.load(feat_path)

        missing = [c for c in feats if c not in X_scaled.columns]
        if missing:
            raise KeyError(
                f"Missing {len(missing)} feature columns for "
                f"{os.path.basename(pair_dir)}/{model_name}_top{n_features}. "
                f"Example missing: {missing[:10]}"
            )

        X_sel = X_scaled[feats]
        preds_by_model.append(model.predict(X_sel))

        # probability totals require predict_proba
        if hasattr(model, "predict_proba"):
            probas_by_model.append(model.predict_proba(X_sel))

    y_pred = majority_vote(preds_by_model)

    count_A = int(np.sum(y_pred == isolate_A))
    count_B = int(np.sum(y_pred == isolate_B))
    N = int(len(y_pred))

    out = {
        "isolate_A": isolate_A,
        "isolate_B": isolate_B,
        "count_A": count_A,
        "count_B": count_B,
        "N": N,
        "model_dir": os.path.basename(pair_dir),
    }

    # Optional: summed probabilities (if available)
    if probas_by_model:
        # Use model.classes_ ordering; assume consistent across the 5 models
        class_names = list(getattr(joblib.load(os.path.join(pair_dir, f"{model_specs[0][0]}_top{model_specs[0][1]}.pkl")), "classes_"))
        totals = class_probability_totals(probas_by_model, class_names)
        out["summed_proba_A"] = float(totals.get(isolate_A, np.nan))
        out["summed_proba_B"] = float(totals.get(isolate_B, np.nan))

    return out


def run_coculture_inference(r: int) -> pd.DataFrame:
    nn = f"{r:02d}"
    models_root = f"round{nn}_models"

    scaler = joblib.load(os.path.join(models_root, "scaler.pkl"))
    log_offsets = joblib.load(os.path.join(models_root, "log_offsets.pkl"))
    numeric_cols = joblib.load(os.path.join(models_root, "numeric_cols.pkl"))
    engineered_cols = joblib.load(os.path.join(models_root, "engineered_feature_columns.pkl"))

    pair_wells = load_pair_wells_index(r)

    if pair_wells.empty:
        print("Warning: no co-culture wells found for this round.")
        return pd.DataFrame()

    out = []
    for _, rec in pair_wells.iterrows():
        df_raw = pd.read_csv(rec["filepath"])

        X_eng = feature_engineer_raw(df_raw, numeric_cols, log_offsets)
        X_eng = X_eng[engineered_cols]
        X_scaled = pd.DataFrame(
            scaler.transform(X_eng), columns=engineered_cols, index=X_eng.index
        )

        res = apply_pair_model_to_well(
            X_scaled=X_scaled,
            isolate_A=rec["IsolateA"],
            isolate_B=rec["IsolateB"],
            models_root=models_root,
        )

        res.update({
            "Round": rec["Round"],
            "Temp": rec["Temp"],
            "Day": rec["Day"],
            "Date": rec["Date"],
            "Well": rec["Well"],
            "Sample": rec["Sample"],
            "Community": rec["Community"],
            "filename": rec["filename"],
        })
        out.append(res)

    df_out = pd.DataFrame(out)

    id_cols = [
        "Round", "Temp", "Day", "Date", "Well", "Sample",
        "isolate_A", "isolate_B", "count_A", "count_B"
    ]
    extra_cols = [c for c in df_out.columns if c not in id_cols]
    df_out = df_out[id_cols + extra_cols]

    key_cols = ["Round", "Temp", "Day", "Date", "Well", "Sample"]
    if df_out.duplicated(subset=key_cols).any():
        print("Warning: duplicate well keys found. Keeping rows separate; consider aggregating.")

    return df_out


def main():
    parser = argparse.ArgumentParser(
        description="Apply saved pairwise ensemble models to co-culture wells."
    )
    parser.add_argument(
        "-r", "--round", type=int, required=True,
        help="Experimental round number (e.g. 1, 2)."
    )
    args = parser.parse_args()

    r = args.round
    df = run_coculture_inference(r)

    out_csv = f"coculture_pairwise_counts_round{r}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}  (rows={len(df)})")
    if not df.empty:
        print(df.head())


if __name__ == "__main__":
    main()
