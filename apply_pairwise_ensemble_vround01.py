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

def build_csv_index(round01_dir="../data/round01"):
    rows = []
    for folder in os.listdir(round01_dir):
        if not folder.endswith("2026"):
            continue
        folder_path = os.path.join(round01_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(".csv"):
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

def load_pair_wells_index(
    metadata_path="../data/round01/all_metadata.csv",
    pairs_to_wells_path="../data/round01/pairs_to_wells.csv",
    round01_dir="../data/round01",
):
    metadata = pd.read_csv(metadata_path)
    pair_meta = metadata[metadata["Type"] == "pair"].copy()

    pairs_to_wells = pd.read_csv(pairs_to_wells_path)
    pairs_to_wells["Round"] = pairs_to_wells["Round"].apply(normalize_round_str)
    pairs_to_wells["Day"] = pairs_to_wells["Day"].astype(str).str.zfill(2)
    pairs_to_wells["Day"] = "d" + pairs_to_wells["Day"].str[-2:]
    pairs_to_wells["Well"] = pairs_to_wells["Well"].astype(str).str.strip()

    csv_index = build_csv_index(round01_dir)

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

def feature_engineer_raw(df_raw, numeric_cols, log_offsets):
    X = df_raw[numeric_cols].copy()
    for col in numeric_cols:
        off = log_offsets[col]
        X[f"{col}_log"] = np.log1p(np.clip(X[col] + off, 0, None))
        X[f"{col}_sqrt"] = np.sqrt(np.clip(X[col], 0, None))
    return X

def majority_vote(preds_by_model):
    mat = np.vstack(preds_by_model)  # (n_models, n_rows)
    out = []
    for j in range(mat.shape[1]):
        vals, counts = np.unique(mat[:, j], return_counts=True)
        out.append(vals[np.argmax(counts)])
    return np.array(out)

def resolve_pair_dir(models_root, isolate_A, isolate_B):
    d1 = os.path.join(models_root, f"{isolate_A}_vs_{isolate_B}")
    d2 = os.path.join(models_root, f"{isolate_B}_vs_{isolate_A}")
    if os.path.isdir(d1):
        return d1
    if os.path.isdir(d2):
        return d2
    return None

def apply_pair_model_to_well(X_scaled, isolate_A, isolate_B, models_root="round01_models"):
    pair_dir = resolve_pair_dir(models_root, isolate_A, isolate_B)
    if pair_dir is None:
        raise FileNotFoundError(f"No model directory found for {isolate_A} vs {isolate_B} under {models_root}/")

    meta = joblib.load(os.path.join(pair_dir, "ensemble_metadata.pkl"))
    model_specs = meta["models"]  # list of (name, n_features)

    preds_by_model = []
    for model_name, n_features in model_specs:
        model_path = os.path.join(pair_dir, f"{model_name}_top{n_features}.pkl")
        feat_path = os.path.join(pair_dir, f"{model_name}_top{n_features}_features.pkl")

        model = joblib.load(model_path)
        feats = joblib.load(feat_path)

        missing = [c for c in feats if c not in X_scaled.columns]
        if missing:
            raise KeyError(
                f"Missing {len(missing)} feature columns for {os.path.basename(pair_dir)}/{model_name}_top{n_features}. "
                f"Example missing: {missing[:10]}"
            )

        preds_by_model.append(model.predict(X_scaled[feats]))

    y_pred = majority_vote(preds_by_model)

    count_A = int(np.sum(y_pred == isolate_A))
    count_B = int(np.sum(y_pred == isolate_B))
    N = int(len(y_pred))

    # simple uncertainty proxy
    pA = count_A / N if N else np.nan
    se_pA = np.sqrt(pA * (1 - pA) / N) if N else np.nan
    se_count_A = se_pA * N if N else np.nan
    se_count_B = se_count_A

    return {
        "isolate_A": isolate_A,
        "isolate_B": isolate_B,
        "count_A": count_A,
        "count_B": count_B,
        "N": N,
        "p_A": pA,
        "se_p_A": se_pA,
        "se_count_A": se_count_A,
        "se_count_B": se_count_B,
        "model_dir": os.path.basename(pair_dir),
    }

def run_coculture_inference_round01(
    models_root="round01_models",
    metadata_path="../data/round01/all_metadata.csv",
    pairs_to_wells_path="../data/round01/pairs_to_wells.csv",
    round01_dir="../data/round01",
):
    scaler = joblib.load(os.path.join(models_root, "round01_scaler.pkl"))
    log_offsets = joblib.load(os.path.join(models_root, "round01_log_offsets.pkl"))
    numeric_cols = joblib.load(os.path.join(models_root, "round01_numeric_cols.pkl"))
    engineered_cols = joblib.load(os.path.join(models_root, "round01_engineered_feature_columns.pkl"))

    pair_wells = load_pair_wells_index(metadata_path, pairs_to_wells_path, round01_dir)

    out = []
    for _, rec in pair_wells.iterrows():
        df_raw = pd.read_csv(rec["filepath"])

        # engineer & scale exactly like training
        X_eng = feature_engineer_raw(df_raw, numeric_cols, log_offsets)
        X_eng = X_eng[engineered_cols]  # enforce exact order
        X_scaled = pd.DataFrame(scaler.transform(X_eng), columns=engineered_cols, index=X_eng.index)

        res = apply_pair_model_to_well(
            X_scaled=X_scaled,
            isolate_A=rec["IsolateA"],
            isolate_B=rec["IsolateB"],
            models_root=models_root,
        )

        # one row per well identifiers
        res.update({
            "Round": rec["Round"],
            "Temp": rec["Temp"],
            "Day": rec["Day"],
            "Date": rec["Date"],
            "Well": rec["Well"],
            "Sample": rec["Sample"],
            "Community": rec["Community"],
            "filename": rec["filename"],
            "filepath": rec["filepath"],
        })
        out.append(res)

    # Ensure one row per well/sample key
    df_out = pd.DataFrame(out)
    key_cols = ["Round", "Temp", "Day", "Date", "Well", "Sample"]
    if df_out.duplicated(subset=key_cols).any():
        # If duplicates exist, keep them separate by filename (or aggregate if you prefer)
        print("Warning: duplicate well keys found. Keeping rows separate; consider aggregating.")
    return df_out

def main():
    df = run_coculture_inference_round01()
    df.to_csv("coculture_pairwise_counts_round01.csv", index=False)
    print(f"Saved coculture_pairwise_counts_round01.csv (rows={len(df)})")
    print(df.head())

if __name__ == "__main__":
    main()
