from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# -------------------------
# Reproducibility (seed=51)
# -------------------------
SEED_DEFAULT = 51


@dataclass(frozen=True)
class SplitFold:
    """
    A single fold split using subject names (trial ids).
    """
    train_subjects: List[str]
    test_subjects: List[str]


TASK_MAP: Dict[str, str] = {
    "knottying": "Knot_Tying",
    "needle_passing": "Needle_Passing",
    "suturing": "Suturing",
}


def set_seed(seed: int) -> None:
    """
    Set random seeds across numpy and tensorflow for reproducibility.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)


def create_model(input_shape: Tuple[int, int], learning_rate: float = 2e-4) -> Sequential:
    """
    Create 1D CNN regression model.

    Args:
        input_shape: (timesteps, channels)
        learning_rate: Adam learning rate

    Returns:
        Compiled Keras model.
    """
    model = Sequential(
        [
            Conv1D(64, 3, activation="selu", padding="same", input_shape=input_shape, kernel_regularizer=l2(1e-5)),
            MaxPooling1D(2),
            Conv1D(128, 3, activation="selu", padding="same", kernel_regularizer=l2(1e-5)),
            MaxPooling1D(2),
            Conv1D(256, 3, activation="selu", padding="same", kernel_regularizer=l2(1e-5)),
            MaxPooling1D(2),
            Flatten(),
            Dense(1024, activation="selu"),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate), loss="mean_squared_error", metrics=["mean_absolute_error"])
    return model


def load_precalc_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load a pre-calculated features CSV.

    Expected columns include:
      subject, grs_score, Flattened_Sequences, duration_seconds, subject_id
      plus feature columns.

    Args:
        csv_path: path to CSV

    Returns:
        DataFrame
    """
    df = pd.read_csv(csv_path)
    if "subject" not in df.columns:
        raise ValueError("CSV must contain 'subject' column.")
    if "grs_score" not in df.columns:
        raise ValueError("CSV must contain 'grs_score' column.")
    return df


def _parse_flattened_sequences(x) -> List[float]:
    """
    Parse Flattened_Sequences cell into a python list.
    Supports:
      - list already
      - string like "[0,1,2,...]"
      - string like "0 1 2 ..."
    """
    if isinstance(x, list):
        return x
    if isinstance(x, (np.ndarray, tuple)):
        return list(x)
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    # Try python literal
    if s.startswith("[") and s.endswith("]"):
        try:
            v = ast.literal_eval(s)
            return list(v) if isinstance(v, (list, tuple, np.ndarray)) else []
        except Exception:
            return []
    # Try space-separated fallback
    try:
        return [float(t) for t in s.replace(",", " ").split()]
    except Exception:
        return []


def expand_event_sequence(df: pd.DataFrame, col: str = "Flattened_Sequences") -> pd.DataFrame:
    """
    Expand Flattened_Sequences into seq_0..seq_L columns (post padded with 0).

    Args:
        df: input df
        col: column holding the flattened sequence

    Returns:
        df with seq_ columns appended and original column dropped.
    """
    seqs = df[col].apply(_parse_flattened_sequences)
    max_len = int(seqs.apply(len).max()) if len(seqs) else 0
    padded = seqs.apply(lambda s: s + [0.0] * max(0, max_len - len(s)))
    seq_df = pd.DataFrame(padded.tolist(), index=df.index).add_prefix("seq_")
    out = pd.concat([df.drop(columns=[col]), seq_df], axis=1)
    return out


def pick_feature_columns(df: pd.DataFrame, modalities: Sequence[str]) -> List[str]:
    """
    Select feature columns from df based on modality rules:
      - events: seq_*
      - kinematics: columns containing 'Left_' or 'Right_' (your indexed kinematics like 0_Left_X etc)
      - slowfast: columns containing 'feature_'
      - density: columns containing 'density_feat_'
      - duration: duration_seconds

    Args:
        df: dataframe with expanded seq_ columns
        modalities: selected modalities

    Returns:
        list of column names to use as X
    """
    cols: List[str] = []

    if "events" in modalities:
        cols.extend([c for c in df.columns if c.startswith("seq_")])

    if "kinematics" in modalities:
        cols.extend([c for c in df.columns if ("Left_" in c) or ("Right_" in c)])

    if "slowfast" in modalities:
        cols.extend([c for c in df.columns if "feature_" in c])

    if "density" in modalities:
        cols.extend([c for c in df.columns if "density_feat_" in c])

    if "duration" in modalities:
        if "duration_seconds" in df.columns:
            cols.append("duration_seconds")

    # de-dup, keep order
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def load_predefined_kfold4(split_json: Path, task_full: str) -> List[SplitFold]:
    """
    Load predefined 4-fold split from a JSON file.

    Expected JSON formats supported:
      1) { "Suturing": [ [test_subjects...], [..], [..], [..] ] }
      2) { "folds": [ {"test": [...]} , ... ] }  (task assumed already filtered externally)

    This returns folds as train/test subject-name lists.

    Args:
        split_json: path to json
        task_full: "Knot_Tying" etc.

    Returns:
        list of SplitFold
    """
    data = json.loads(Path(split_json).read_text())
    if task_full in data and isinstance(data[task_full], list):
        test_folds = data[task_full]
    elif "folds" in data:
        test_folds = [f["test"] for f in data["folds"]]
    else:
        raise ValueError(f"Unsupported split json format: {split_json}")

    folds: List[SplitFold] = []
    # Build full subject universe from provided folds
    all_subj = set()
    for test_subj in test_folds:
        for s in test_subj:
            all_subj.add(str(s))

    for test_subj in test_folds:
        test_list = [str(s) for s in test_subj]
        train_list = sorted(list(all_subj - set(test_list)))
        folds.append(SplitFold(train_subjects=train_list, test_subjects=test_list))
    return folds


def build_louo_folds(df: pd.DataFrame, group_col: str = "subject_id") -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Build Leave-One-User-Out folds based on subject_id.

    Args:
        df: dataframe containing group_col
        group_col: grouping column

    Returns:
        list of tuples: (test_group, train_mask, test_mask)
    """
    if group_col not in df.columns:
        raise ValueError(f"LOUO requires '{group_col}' column in the CSV.")
    groups = [g for g in df[group_col].dropna().unique().tolist()]
    groups = sorted([str(g) for g in groups])
    folds = []
    for g in groups:
        test_mask = df[group_col].astype(str) == g
        train_mask = ~test_mask
        folds.append((g, train_mask.values, test_mask.values))
    return folds


def run_kfold4_experiment(
    df: pd.DataFrame,
    task_full: str,
    modalities: Sequence[str],
    split_json: Path,
    out_dir: Path,
    seed: int,
    pca_variance: float = 0.99,
) -> None:
    """
    Run predefined kfold4 experiment using subject-name folds.

    Saves:
      - per-fold model (.h5)
      - per-fold predictions CSV
      - overall metrics CSV
      - overall predictions CSV

    Args:
        df: prepared dataframe
        task_full: task name
        modalities: selected modalities
        split_json: predefined split json
        out_dir: output directory
        seed: random seed
        pca_variance: PCA retained variance
    """
    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Expand events
    if "Flattened_Sequences" in df.columns:
        df = expand_event_sequence(df, col="Flattened_Sequences")

    feature_cols = pick_feature_columns(df, modalities)
    if not feature_cols:
        raise ValueError(f"No features selected. modalities={modalities}")

    X_all = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_all = df["grs_score"].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    subjects = df["subject"].astype(str)

    folds = load_predefined_kfold4(split_json, task_full)

    fold_metrics: List[Dict] = []
    y_results = []

    for fold_idx, fold in enumerate(folds, start=1):
        train_mask = subjects.isin(fold.train_subjects)
        test_mask = subjects.isin(fold.test_subjects)

        X_train, X_test = X_all[train_mask], X_all[test_mask]
        y_train, y_test = y_all[train_mask], y_all[test_mask]
        test_names = subjects[test_mask].values

        # scale + pca
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        pca = PCA(n_components=float(pca_variance))
        X_train_p = pca.fit_transform(X_train_s)
        X_test_p = pca.transform(X_test_s)

        # scale target
        y_scaler = MinMaxScaler()
        y_train_s = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test_s = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

        # Conv1D expects (N, T, C)
        X_train_p = X_train_p[..., np.newaxis]
        X_test_p = X_test_p[..., np.newaxis]

        model = create_model((X_train_p.shape[1], 1))
        es = EarlyStopping(monitor="val_loss", patience=200, restore_best_weights=True)

        model.fit(
            X_train_p,
            y_train_s,
            validation_data=(X_test_p, y_test_s),
            epochs=1000,
            batch_size=1,
            callbacks=[es],
            verbose=2,
        )

        y_pred_s = model.predict(X_test_p).flatten()
        y_pred = y_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
        y_true = y_scaler.inverse_transform(y_test_s.reshape(-1, 1)).flatten()

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        sp, _ = spearmanr(y_true, y_pred)

        fold_metrics.append(
            {
                "Fold": fold_idx,
                "MSE": mse,
                "RMSE": float(np.sqrt(mse)),
                "MAE": mae,
                "R2": r2,
                "Spearman": float(sp),
                "PCA_components": int(X_train_p.shape[1]),
                "Modalities": ",".join(modalities),
            }
        )

        fold_pred_df = pd.DataFrame(
            {"Fold": fold_idx, "subject": test_names, "y_true": y_true, "y_pred": y_pred}
        )
        y_results.append(fold_pred_df)

        # Save fold artifacts
        model_path = out_dir / f"model_{task_full}_kfold4_fold{fold_idx}_seed{seed}.h5"
        model.save(model_path)

        preds_path = out_dir / f"preds_{task_full}_kfold4_fold{fold_idx}_seed{seed}.csv"
        fold_pred_df.to_csv(preds_path, index=False)

    metrics_df = pd.DataFrame(fold_metrics)
    all_preds_df = pd.concat(y_results, ignore_index=True) if y_results else pd.DataFrame()

    metrics_df.to_csv(out_dir / f"metrics_{task_full}_kfold4_seed{seed}.csv", index=False)
    all_preds_df.to_csv(out_dir / f"preds_{task_full}_kfold4_seed{seed}.csv", index=False)

    print(metrics_df.round(3))
    print(all_preds_df.head())


def run_louo_experiment(
    df: pd.DataFrame,
    task_full: str,
    modalities: Sequence[str],
    out_dir: Path,
    seed: int,
    pca_variance: float = 0.99,
) -> None:
    """
    Run Leave-One-User-Out experiment using subject_id.

    Args:
        df: prepared dataframe
        task_full: task name
        modalities: selected modalities
        out_dir: output directory
        seed: random seed
        pca_variance: PCA retained variance
    """
    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "Flattened_Sequences" in df.columns:
        df = expand_event_sequence(df, col="Flattened_Sequences")

    feature_cols = pick_feature_columns(df, modalities)
    if not feature_cols:
        raise ValueError(f"No features selected. modalities={modalities}")

    X_all = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_all = df["grs_score"].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    subjects = df["subject"].astype(str)

    louo = build_louo_folds(df, group_col="subject_id")

    fold_metrics: List[Dict] = []
    y_results = []

    for fold_idx, (test_group, train_mask, test_mask) in enumerate(louo, start=1):
        X_train, X_test = X_all.iloc[train_mask], X_all.iloc[test_mask]
        y_train, y_test = y_all.iloc[train_mask], y_all.iloc[test_mask]
        test_names = subjects.iloc[test_mask].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        pca = PCA(n_components=float(pca_variance))
        X_train_p = pca.fit_transform(X_train_s)
        X_test_p = pca.transform(X_test_s)

        y_scaler = MinMaxScaler()
        y_train_s = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test_s = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

        X_train_p = X_train_p[..., np.newaxis]
        X_test_p = X_test_p[..., np.newaxis]

        model = create_model((X_train_p.shape[1], 1))
        es = EarlyStopping(monitor="val_loss", patience=100, restore_best_weights=True)

        model.fit(
            X_train_p,
            y_train_s,
            validation_data=(X_test_p, y_test_s),
            epochs=500,
            batch_size=1,
            callbacks=[es],
            verbose=2,
        )

        y_pred_s = model.predict(X_test_p).flatten()
        y_pred = y_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
        y_true = y_scaler.inverse_transform(y_test_s.reshape(-1, 1)).flatten()

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        sp, _ = spearmanr(y_true, y_pred)

        fold_metrics.append(
            {
                "Fold": fold_idx,
                "Test_Subject_ID": str(test_group),
                "MSE": mse,
                "RMSE": float(np.sqrt(mse)),
                "MAE": mae,
                "R2": r2,
                "Spearman": float(sp),
                "PCA_components": int(X_train_p.shape[1]),
                "Modalities": ",".join(modalities),
            }
        )

        fold_pred_df = pd.DataFrame(
            {"Fold": fold_idx, "subject": test_names, "y_true": y_true, "y_pred": y_pred}
        )
        y_results.append(fold_pred_df)

        model_path = out_dir / f"model_{task_full}_louo_fold{fold_idx}_seed{seed}.h5"
        model.save(model_path)

        preds_path = out_dir / f"preds_{task_full}_louo_fold{fold_idx}_seed{seed}.csv"
        fold_pred_df.to_csv(preds_path, index=False)

    metrics_df = pd.DataFrame(fold_metrics)
    all_preds_df = pd.concat(y_results, ignore_index=True) if y_results else pd.DataFrame()

    metrics_df.to_csv(out_dir / f"metrics_{task_full}_louo_seed{seed}.csv", index=False)
    all_preds_df.to_csv(out_dir / f"preds_{task_full}_louo_seed{seed}.csv", index=False)

    print(metrics_df.round(3))
    print(all_preds_df.head())


def main() -> None:
    """
    CLI entrypoint.

    Example:
      python src/train.py --task suturing --split kfold4 --modalities events kinematics --seed 51
    """
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--csv_root", type=str, default="data/pre_calculated_features", help="Folder with *_all_features_combined.csv")
    p.add_argument("--task", type=str, required=True, choices=list(TASK_MAP.keys()))
    p.add_argument("--split", type=str, required=True, choices=["kfold4", "louo"])
    p.add_argument("--modalities", type=str, nargs="+", required=True, choices=["events", "kinematics", "slowfast", "density", "duration"])
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--pca_variance", type=float, default=0.99)
    p.add_argument("--out_root", type=str, default="runs_precalc")
    p.add_argument("--split_json", type=str, default="splits/kfold4.json", help="Used only for kfold4")
    args = p.parse_args()

    task_full = TASK_MAP[args.task]

    csv_name = {
        "Knot_Tying": "knottying_all_features_combined.csv",
        "Needle_Passing": "needle_passing_all_features_combined.csv",
        "Suturing": "suturing_all_features_combined.csv",
    }[task_full]

    csv_path = Path(args.csv_root) / csv_name
    df = load_precalc_csv(csv_path)

    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    mod_tag = "-".join(args.modalities)
    out_dir = Path(args.out_root) / f"{task_full}_{args.split}_{mod_tag}_seed{int(args.seed)}_{run_id}"

    out_dir.mkdir(parents=True, exist_ok=True)

    # save config snapshot
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "task": task_full,
                "split": args.split,
                "modalities": args.modalities,
                "seed": int(args.seed),
                "pca_variance": float(args.pca_variance),
                "csv_path": str(csv_path),
            },
            indent=2,
        )
    )

    if args.split == "kfold4":
        run_kfold4_experiment(
            df=df,
            task_full=task_full,
            modalities=args.modalities,
            split_json=Path(args.split_json),
            out_dir=out_dir,
            seed=int(args.seed),
            pca_variance=float(args.pca_variance),
        )
    else:
        run_louo_experiment(
            df=df,
            task_full=task_full,
            modalities=args.modalities,
            out_dir=out_dir,
            seed=int(args.seed),
            pca_variance=float(args.pca_variance),
        )


if __name__ == "__main__":
    main()
