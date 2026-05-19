# rf_full_holdout_parametric.py
# ------------------------------------------------------------
# Random Forest regression for DF_bulk or DF_seg with:
#  - Input CSV dataset
#  - Fixed feature set (substitutional elements only)
#  - Missing compositional features filled with 0
#  - Missing target rows removed
#  - Reproducible random 80/20 split
#  - Hyperparameter tuning on train80 only via internal CV
#  - Final fit on full train80 with best hyperparameters
#  - Final evaluation on sealed test20
#  - CV variability diagnostics on train80
#  - Learning curves:
#       (1) CV learning curve on train80
#       (2) Holdout learning curve using test20
#  - Residual diagnostics on test20
#  - OOB diagnostics on train80
#  - Model persistence with joblib
#  - Saving raw tables used for figures
#
# Edit CONFIG only.
# ------------------------------------------------------------

from __future__ import annotations

import gc
import hashlib
import json
import math
import traceback
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RepeatedKFold, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.exceptions import TrialPruned


# =========================
# CONFIG (EDIT HERE)
# =========================

SEED = 42

# ---- target ----
TARGET = "DF_seg"

# ---- input ----
DATA_PATH = Path(r"C:\Users\dwack\Documents\Python Scripts\sh_ga\ml_df\woking_dataset.csv")

# ---- outputs ----
OUT_DIR = Path(r"C:\Users\dwack\Documents\Python Scripts\sh_ga\ml_df\output_rf_df_seg")

# ---- columns ----
ID_COL = "alloy_id"
FEATURE_COLS = ["Mn", "Si", "Cr", "Mo", "Ni", "Cu", "Co", "V", "Nb", "W", "Al", "Ti"]
EXCLUDED_INTERSTITIALS = ["C", "B", "N"]
OPTIONAL_IGNORE_COLS = [
    "Cluster", "DF_seg", "DF_bulk", "IE_bulk", "diff_Mo", "diff_W",
    "TD_Fe", "real_fake", "SURF_ENERGY", "is_real"
]

# ---- split ----
TEST_SIZE = 0.20

# ---- tuning ----
N_TRIALS = 240
OPTUNA_N_JOBS = 8
TUNE_ESTIMATORS_MIN = 300
TUNE_ESTIMATORS_MAX = 600
FINAL_N_ESTIMATORS = 600
N_JOBS_RF = -1

# ---- CV variability (fixed params, train80 only) ----
DO_CV_VARIABILITY = True
RK_N_SPLITS = 5
RK_N_REPEATS = 10

# ---- NRMSE denominator ----
NRMSE_DEN_MODE = "fold"  # "fold" or "global"

# ---- learning curves (CV on train80) ----
DO_LEARNING_CURVES_CV = True
LC_CV_FRACTIONS = [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0]
LC_CV_N_SPLITS = 5
LC_CV_MAX_POINTS = 5000
LC_CV_MIN_POINTS = 50

# ---- learning curve (holdout: train subsets vs test20) ----
DO_LEARNING_CURVE_HOLDOUT = True
LC_HOLDOUT_TRAIN_SIZES = [0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 1.00]
LC_HOLDOUT_REPEATS = 10

# ---- residuals / OOB / PI ----
DO_RESIDUAL_DIAGNOSTICS = True
DO_OOB_DIAGNOSTICS = True
DO_PERMUTATION_IMPORTANCE_TEST20 = True
PI_N_REPEATS = 10


# =========================
# Helpers
# =========================

def derive_u32(*parts) -> int:
    h = hashlib.blake2b(digest_size=4)
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest(), "big")


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def sanitize(o):
    if isinstance(o, dict):
        return {str(k): sanitize(v) for k, v in o.items()}
    if isinstance(o, list):
        return [sanitize(v) for v in o]
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def file_fingerprint(path: Path, n_bytes: int = 2_000_000) -> Dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(n_bytes))
    return {
        "path": str(path),
        "exists": True,
        "size": int(st.st_size),
        "mtime": float(st.st_mtime),
        "sha256_head": h.hexdigest(),
        "bytes_hashed": int(n_bytes),
    }


def save_json(obj: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def save_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def metrics_dict(y_true, y_pred, nrmse_den_sd: Optional[float] = None) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    _rmse = rmse(y_true, y_pred)
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))

    if nrmse_den_sd is None or (not np.isfinite(nrmse_den_sd)) or nrmse_den_sd <= 0:
        sd = float(np.std(y_true, ddof=1))
    else:
        sd = float(nrmse_den_sd)

    nrmse_val = float(_rmse / sd) if sd > 0 else float("nan")

    return {
        "RMSE": float(_rmse),
        "MAE": mae,
        "R2": r2,
        "NRMSE": nrmse_val,
        "ME_bias": bias,
    }


def validate_target(target: str):
    if target not in {"DF_bulk", "DF_seg"}:
        raise ValueError(f"TARGET must be 'DF_bulk' or 'DF_seg'. Got: {target}")


def validate_required_columns(df: pd.DataFrame, cols: List[str], where: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {where}: {missing}")


def coerce_numeric_columns(df: pd.DataFrame, cols: List[str], where: str) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    bad = [c for c in cols if out[c].isna().all()]
    if bad:
        raise ValueError(f"Columns became fully NaN after numeric coercion in {where}: {bad}")
    return out


# =========================
# Data loading / prep
# =========================

def load_dataset_for_target(
    data_path: Path,
    target: str,
    feature_cols: List[str],
    id_col: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, Dict[str, Any]]:
    if not data_path.exists():
        raise FileNotFoundError(str(data_path))

    df = pd.read_csv(data_path)
    original_n = len(df)

    validate_required_columns(df, feature_cols + [target], where="input CSV")

    # ID column is optional
    has_id = id_col in df.columns
    if has_id:
        ids = df[id_col].astype(str).copy()
    else:
        ids = pd.Series([f"row_{i}" for i in range(len(df))], name=id_col, dtype=str)

    # Keep only the requested features
    X = df[feature_cols].copy()

    # Numeric coercion
    X = coerce_numeric_columns(X, feature_cols, where="features")
    y = pd.to_numeric(df[target], errors="coerce")

    # Missing compositions -> 0
    X = X.fillna(0.0)

    # Missing target -> drop row
    mask = y.notna().values
    dropped_target_nan = int((~mask).sum())

    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    ids = ids.loc[mask].reset_index(drop=True)
    df_used = df.loc[mask].reset_index(drop=True)

    info = {
        "n_rows_original": int(original_n),
        "n_rows_used": int(len(y)),
        "n_rows_dropped_target_nan": int(dropped_target_nan),
        "has_id_column": bool(has_id),
        "feature_cols_used": list(feature_cols),
        "excluded_interstitials": list(EXCLUDED_INTERSTITIALS),
    }

    if len(y) < 30:
        raise ValueError(f"Too few usable rows after dropping missing target: n={len(y)}")

    return X, y, ids, df_used, info


# =========================
# Tuning (train80 only)
# =========================

def optuna_tune_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
    n_trials: int,
    optuna_n_jobs: int,
    tune_estimators_min: int,
    tune_estimators_max: int,
) -> Dict[str, Any]:
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", tune_estimators_min, tune_estimators_max),
            "max_depth": trial.suggest_int("max_depth", 10, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 80),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 40),
            "max_features": trial.suggest_float("max_features", 0.1, 0.9),
            "bootstrap": True,
            "max_samples": trial.suggest_float("max_samples", 0.3, 1.0),
            "oob_score": False,
        }

        rmses: List[float] = []
        for fold, (tr_idx, va_idx) in enumerate(splitter.split(X_train), 1):
            Xt, Xv = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            yt, yv = y_train.iloc[tr_idx], y_train.iloc[va_idx]

            m = RandomForestRegressor(
                **params,
                random_state=derive_u32(seed, "tune_fold", fold),
                n_jobs=1,
            )
            m.fit(Xt, yt)
            pred = m.predict(Xv)
            rmses.append(rmse(yv, pred))

            trial.report(float(np.mean(rmses)), step=fold)
            if trial.should_prune() and fold >= 3:
                raise TrialPruned()

        return float(np.mean(rmses))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_warmup_steps=3),
    )

    if n_trials <= 0:
        raise ValueError("n_trials must be > 0")

    with tqdm(total=n_trials, desc="Optuna (train80 CV tuning)", ncols=95) as pbar:
        def _cb(study, trial):
            pbar.update(1)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False, callbacks=[_cb], n_jobs=optuna_n_jobs)

    return dict(study.best_params)


# =========================
# CV variability (train80 only)
# =========================

def cv_variability_train80(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    params: Dict[str, Any],
    rk_n_splits: int,
    rk_n_repeats: int,
    seed: int,
    n_jobs_rf: int,
    nrmse_den_mode: str,
    out_dir: Path,
):
    rk = RepeatedKFold(
        n_splits=rk_n_splits,
        n_repeats=rk_n_repeats,
        random_state=derive_u32(seed, "rk")
    )
    records = []

    sd_global = float(np.std(y_tr.values.astype(float), ddof=1)) if nrmse_den_mode == "global" else None

    fold_id = 0
    total_folds = rk_n_splits * rk_n_repeats
    for tr_idx, te_idx in tqdm(rk.split(X_tr), total=total_folds, desc="CV variability (train80)", ncols=95):
        fold_id += 1
        Xt, Xv = X_tr.iloc[tr_idx], X_tr.iloc[te_idx]
        yt, yv = y_tr.iloc[tr_idx], y_tr.iloc[te_idx]

        den = sd_global
        if nrmse_den_mode == "fold":
            den = float(np.std(yt.values.astype(float), ddof=1)) if len(yt) > 1 else sd_global

        m = RandomForestRegressor(
            **params,
            random_state=derive_u32(seed, "cv", fold_id),
            n_jobs=n_jobs_rf
        )
        m.fit(Xt, yt)
        pv = m.predict(Xv)

        mm = metrics_dict(yv, pv, nrmse_den_sd=den)
        mm["fold_id"] = int(fold_id)
        mm["n_test"] = int(len(yv))
        records.append(mm)

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "cv_train80_scores.csv", index=False)

    summary = df[["RMSE", "MAE", "R2", "NRMSE", "ME_bias"]].agg(["mean", "std"]).to_dict()
    save_json(
        {
            "cv_train80_summary": summary,
            "n_splits": rk_n_splits,
            "n_repeats": rk_n_repeats
        },
        out_dir / "cv_train80_summary.json"
    )


# =========================
# Learning curves
# =========================

def run_learning_curves_cv_train80(
    X_train80: pd.DataFrame,
    y_train80: pd.Series,
    model_params: Dict[str, Any],
    out_dir: Path,
    seed: int,
    fractions: List[float],
    n_splits: int,
    max_points: int,
    min_points: int,
    n_jobs_rf: int,
):
    rng = np.random.RandomState(derive_u32(seed, "learning_curve_cv_subsample"))
    n_total = len(X_train80)
    if n_total < max(min_points, n_splits * 2):
        raise ValueError(f"Not enough train80 points for CV learning curves: n={n_total}")

    results = []
    X = X_train80.reset_index(drop=True)
    y = y_train80.reset_index(drop=True)

    for f in fractions:
        f = float(f)
        if not (0 < f <= 1.0):
            continue

        n_sub = int(round(f * n_total))
        n_sub = max(min_points, n_sub)
        n_sub = min(n_sub, n_total)
        n_sub = min(n_sub, max_points)

        idx = rng.choice(np.arange(n_total), size=n_sub, replace=False)
        Xs = X.iloc[idx].reset_index(drop=True)
        ys = y.iloc[idx].reset_index(drop=True)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=derive_u32(seed, "learning_curve_cv_kfold", f))
        tr_scores = []
        va_scores = []

        fold = 0
        for tr_idx, va_idx in kf.split(Xs):
            fold += 1
            Xt, Xv = Xs.iloc[tr_idx], Xs.iloc[va_idx]
            yt, yv = ys.iloc[tr_idx], ys.iloc[va_idx]

            m = RandomForestRegressor(
                **model_params,
                random_state=derive_u32(seed, "learning_curve_cv_fit", f, fold),
                n_jobs=n_jobs_rf,
            )
            m.fit(Xt, yt)

            p_tr = m.predict(Xt)
            p_va = m.predict(Xv)

            tr_scores.append(rmse(yt, p_tr))
            va_scores.append(rmse(yv, p_va))

        results.append(
            {
                "fraction": f,
                "n_train": int(n_sub),
                "rmse_train_mean": float(np.mean(tr_scores)),
                "rmse_val_mean": float(np.mean(va_scores)),
                "rmse_val_std": float(np.std(va_scores, ddof=1)) if len(va_scores) > 1 else float("nan"),
            }
        )

    df = pd.DataFrame(results).sort_values(["fraction", "n_train"])
    df.to_csv(out_dir / "learning_curve_cv_train80_rmse.csv", index=False)

    plt.figure(figsize=(7.2, 5.2))
    plt.plot(df["n_train"], df["rmse_train_mean"], marker="o", label="Train RMSE")
    plt.plot(df["n_train"], df["rmse_val_mean"], marker="o", label="CV RMSE")
    plt.fill_between(
        df["n_train"].to_numpy(),
        (df["rmse_val_mean"] - df["rmse_val_std"]).to_numpy(),
        (df["rmse_val_mean"] + df["rmse_val_std"]).to_numpy(),
        alpha=0.2,
        linewidth=0.0,
    )
    plt.xlabel("Training subset size (from train80)")
    plt.ylabel("RMSE")
    plt.title(f"CV learning curve (train80) — {TARGET}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "learning_curve_cv_train80_rmse.png", dpi=220)
    plt.close()


def learning_curve_holdout_r2_rmse(
    X_train80: pd.DataFrame,
    y_train80: pd.Series,
    X_test20: pd.DataFrame,
    y_test20: pd.Series,
    model_params: Dict[str, Any],
    seed: int,
    train_sizes: List[float],
    repeats: int,
    n_jobs_rf: int,
    out_dir: Path,
):
    rng = np.random.RandomState(derive_u32(seed, "lc_holdout_rng"))
    n = len(X_train80)
    idx_all = np.arange(n)

    rows = []
    for s in train_sizes:
        if not (0 < s <= 1.0):
            raise ValueError(f"Invalid train_size: {s}")
        n_sub = max(5, int(round(s * n)))
        n_sub = min(n_sub, n)

        for r in range(repeats):
            sel = rng.choice(idx_all, size=n_sub, replace=False)
            Xs = X_train80.iloc[sel]
            ys = y_train80.iloc[sel]

            m = RandomForestRegressor(
                **model_params,
                random_state=derive_u32(seed, "lc_holdout", s, r),
                n_jobs=n_jobs_rf,
            )
            m.fit(Xs, ys)

            pred_tr = m.predict(Xs)
            pred_te = m.predict(X_test20)

            rows.append({
                "train_frac": float(s),
                "n_train": int(n_sub),
                "repeat": int(r),
                "r2_train": float(r2_score(ys, pred_tr)),
                "r2_test20": float(r2_score(y_test20, pred_te)),
                "rmse_train": float(rmse(ys, pred_tr)),
                "rmse_test20": float(rmse(y_test20, pred_te)),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "learning_curve_holdout_raw.csv", index=False)

    agg = df.groupby(["train_frac", "n_train"]).agg(
        r2_train_mean=("r2_train", "mean"),
        r2_train_std=("r2_train", "std"),
        r2_test20_mean=("r2_test20", "mean"),
        r2_test20_std=("r2_test20", "std"),
        rmse_train_mean=("rmse_train", "mean"),
        rmse_train_std=("rmse_train", "std"),
        rmse_test20_mean=("rmse_test20", "mean"),
        rmse_test20_std=("rmse_test20", "std"),
    ).reset_index()
    agg.to_csv(out_dir / "learning_curve_holdout_summary.csv", index=False)

    # R2 plot
    fig = plt.figure(figsize=(7.2, 5.2))
    x = agg["n_train"].to_numpy()
    plt.errorbar(
        x, agg["r2_train_mean"], yerr=agg["r2_train_std"],
        fmt="-o", capsize=3, label="R² train"
    )
    plt.errorbar(
        x, agg["r2_test20_mean"], yerr=agg["r2_test20_std"],
        fmt="-o", capsize=3, label="R² test20"
    )
    plt.xlabel("N training samples (subsample of train80)")
    plt.ylabel("R²")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "learning_curve_holdout_r2.png", dpi=220)
    plt.close(fig)

    # RMSE plot
    fig = plt.figure(figsize=(7.2, 5.2))
    plt.errorbar(
        x, agg["rmse_train_mean"], yerr=agg["rmse_train_std"],
        fmt="-o", capsize=3, label="RMSE train"
    )
    plt.errorbar(
        x, agg["rmse_test20_mean"], yerr=agg["rmse_test20_std"],
        fmt="-o", capsize=3, label="RMSE test20"
    )
    plt.xlabel("N training samples (subsample of train80)")
    plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "learning_curve_holdout_rmse.png", dpi=220)
    plt.close(fig)


# =========================
# Diagnostics
# =========================

def run_residual_diagnostics_test20(
    ids_test20: pd.Series,
    y_true: pd.Series,
    y_pred: np.ndarray,
    out_dir: Path,
    id_col: str,
):
    df = pd.DataFrame(
        {
            id_col: ids_test20.astype(str).values,
            "y_true": y_true.values.astype(float),
            "y_pred": np.asarray(y_pred, dtype=float),
        }
    )
    df["residual"] = df["y_pred"] - df["y_true"]
    df["abs_residual"] = np.abs(df["residual"])
    save_parquet(df, out_dir / "residuals_test20.parquet")

    plt.figure(figsize=(6.6, 5.2))
    plt.scatter(df["y_pred"], df["residual"], s=18, alpha=0.35, edgecolors="none")
    plt.axhline(0.0, color="black", linewidth=1.2, linestyle="--")
    plt.xlabel("y_pred (test20)")
    plt.ylabel("residual = y_pred - y_true")
    plt.title(f"Residuals vs predictions (test20) — {TARGET}")
    plt.tight_layout()
    plt.savefig(out_dir / "resid_vs_yhat_test20.png", dpi=220)
    plt.close()

    plt.figure(figsize=(6.6, 5.2))
    plt.hist(df["residual"].to_numpy(), bins=40, alpha=0.85, edgecolor="black", linewidth=0.6)
    plt.axvline(0.0, color="black", linewidth=1.2, linestyle="--")
    plt.xlabel("residual = y_pred - y_true")
    plt.ylabel("count")
    plt.title(f"Residual distribution (test20) — {TARGET}")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_residuals_test20.png", dpi=220)
    plt.close()


def run_oob_diagnostics_train80(
    X_train80: pd.DataFrame,
    y_train80: pd.Series,
    model_params: Dict[str, Any],
    out_dir: Path,
    seed: int,
    n_jobs_rf: int,
):
    info: Dict[str, Any] = {"enabled": True, "note": ""}

    if not bool(model_params.get("bootstrap", True)):
        info["enabled"] = False
        info["note"] = "OOB skipped because bootstrap=False in model_params."
        save_json(info, out_dir / "oob_diagnostics.json")
        return

    params_oob = dict(model_params)
    params_oob["oob_score"] = True

    m = RandomForestRegressor(
        **params_oob,
        random_state=derive_u32(seed, "oob_fit"),
        n_jobs=n_jobs_rf
    )
    m.fit(X_train80, y_train80)

    info["oob_score_r2"] = float(getattr(m, "oob_score_", float("nan")))

    if hasattr(m, "oob_prediction_") and m.oob_prediction_ is not None:
        oob_pred = np.asarray(m.oob_prediction_, dtype=float)
        yv = y_train80.values.astype(float)
        finite = np.isfinite(oob_pred)
        if finite.sum() > 5:
            info["oob_rmse"] = float(rmse(yv[finite], oob_pred[finite]))
            info["oob_mae"] = float(mean_absolute_error(yv[finite], oob_pred[finite]))
            info["oob_n_used"] = int(finite.sum())
        else:
            info["note"] = "oob_prediction_ had too few finite entries to score."
    else:
        info["note"] = "oob_prediction_ not available in this sklearn version / configuration."

    save_json(info, out_dir / "oob_diagnostics.json")


# =========================
# Main
# =========================

def main():
    validate_target(TARGET)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    save_json(
        {
            "SEED": int(SEED),
            "TARGET": TARGET,
            "DATA_PATH": file_fingerprint(DATA_PATH),
            "FEATURE_COLS": FEATURE_COLS,
            "TEST_SIZE": float(TEST_SIZE),
            "N_TRIALS": int(N_TRIALS),
            "versions": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "sklearn": __import__("sklearn").__version__,
                "optuna": optuna.__version__,
                "joblib": joblib.__version__,
            },
        },
        OUT_DIR / "inputs_fingerprint.json",
    )

    # -------------------------
    # Load and prepare
    # -------------------------
    X_all, y_all, ids_all, df_used, data_info = load_dataset_for_target(
        data_path=DATA_PATH,
        target=TARGET,
        feature_cols=FEATURE_COLS,
        id_col=ID_COL,
    )
    save_json(data_info, OUT_DIR / "data_prep_summary.json")

    # Save cleaned modelling table
    modelling_df = X_all.copy()
    modelling_df.insert(0, ID_COL, ids_all.values)
    modelling_df[TARGET] = y_all.values
    save_parquet(modelling_df, OUT_DIR / "modelling_table_clean.parquet")

    # -------------------------
    # Reproducible 80/20 split
    # -------------------------
    idx = np.arange(len(y_all))
    tr_idx, te_idx = train_test_split(
        idx,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True,
    )

    X_train80 = X_all.iloc[tr_idx].reset_index(drop=True)
    y_train80 = y_all.iloc[tr_idx].reset_index(drop=True)
    ids_train80 = ids_all.iloc[tr_idx].reset_index(drop=True)

    X_test20 = X_all.iloc[te_idx].reset_index(drop=True)
    y_test20 = y_all.iloc[te_idx].reset_index(drop=True)
    ids_test20 = ids_all.iloc[te_idx].reset_index(drop=True)

    save_parquet(
        pd.concat(
            [pd.DataFrame({ID_COL: ids_train80.values}), X_train80, y_train80.rename(TARGET)],
            axis=1
        ),
        OUT_DIR / "train80_with_ids.parquet"
    )
    save_parquet(
        pd.concat(
            [pd.DataFrame({ID_COL: ids_test20.values}), X_test20, y_test20.rename(TARGET)],
            axis=1
        ),
        OUT_DIR / "test20_with_ids.parquet"
    )

    save_json(
        {
            "n_total": int(len(y_all)),
            "n_train80": int(len(y_train80)),
            "n_test20": int(len(y_test20)),
            "test_size": float(TEST_SIZE),
            "split_random_state": int(SEED),
            "split_type": "random_reproducible",
        },
        OUT_DIR / "split_info.json"
    )

    # -------------------------
    # Tuning on train80 only
    # -------------------------
    best_params_path = OUT_DIR / "best_params_train80.json"
    if best_params_path.exists():
        best_params = json.loads(best_params_path.read_text(encoding="utf-8"))
    else:
        best_params = optuna_tune_rf(
            X_train=X_train80,
            y_train=y_train80,
            seed=derive_u32(SEED, "tune_train80"),
            n_trials=N_TRIALS,
            optuna_n_jobs=OPTUNA_N_JOBS,
            tune_estimators_min=TUNE_ESTIMATORS_MIN,
            tune_estimators_max=TUNE_ESTIMATORS_MAX,
        )
        save_json(best_params, best_params_path)

    model_params = dict(best_params)
    if FINAL_N_ESTIMATORS is not None and int(model_params.get("n_estimators", 0)) < int(FINAL_N_ESTIMATORS):
        model_params["n_estimators"] = int(FINAL_N_ESTIMATORS)
    model_params["oob_score"] = False
    if not model_params.get("bootstrap", True):
        model_params["max_samples"] = None

    save_json(
        {
            "best_params_from_train80_cv": best_params,
            "model_params_used_final_fit": model_params,
            "selection_metric": "mean CV RMSE on train80",
        },
        OUT_DIR / "model_params_used.json",
    )

    # -------------------------
    # Final fit on full train80
    # -------------------------
    model = RandomForestRegressor(
        **model_params,
        random_state=derive_u32(SEED, "fit_train80_full"),
        n_jobs=N_JOBS_RF
    )
    model.fit(X_train80, y_train80)

    # Save model
    joblib.dump(model, OUT_DIR / "rf_model.joblib", compress=3)

    # -------------------------
    # Predictions and metrics
    # -------------------------
    yhat_train = model.predict(X_train80)
    yhat_test = model.predict(X_test20)

    sd_train80 = float(np.std(y_train80.values.astype(float), ddof=1)) if len(y_train80) > 1 else float("nan")
    m_train = metrics_dict(y_train80, yhat_train, nrmse_den_sd=sd_train80)
    m_test = metrics_dict(y_test20, yhat_test, nrmse_den_sd=sd_train80)

    pd.DataFrame([m_train]).to_csv(OUT_DIR / "metrics_train80.csv", index=False)
    pd.DataFrame([m_test]).to_csv(OUT_DIR / "metrics_test20.csv", index=False)

    save_parquet(
        pd.DataFrame({
            ID_COL: ids_train80.values,
            "y_true": y_train80.values.astype(float),
            "y_pred": yhat_train.astype(float)
        }),
        OUT_DIR / "pred_train80_with_ids.parquet"
    )
    save_parquet(
        pd.DataFrame({
            ID_COL: ids_test20.values,
            "y_true": y_test20.values.astype(float),
            "y_pred": yhat_test.astype(float)
        }),
        OUT_DIR / "pred_test20_with_ids.parquet"
    )

    # -------------------------
    # Built-in feature importance
    # -------------------------
    fi = pd.DataFrame({
        "feature": X_train80.columns,
        "importance_gini": model.feature_importances_.astype(float)
    }).sort_values("importance_gini", ascending=False)
    fi.to_csv(OUT_DIR / "feature_importance_gini.csv", index=False)

    # -------------------------
    # Permutation importance on test20
    # -------------------------
    if DO_PERMUTATION_IMPORTANCE_TEST20:
        try:
            pi = permutation_importance(
                model,
                X_test20,
                y_test20,
                n_repeats=PI_N_REPEATS,
                random_state=derive_u32(SEED, "pi_test20"),
                n_jobs=N_JOBS_RF,
            )
            df_pi = pd.DataFrame({
                "feature": X_train80.columns,
                "importance_mean": pi.importances_mean.astype(float),
                "importance_std": pi.importances_std.astype(float),
            }).sort_values("importance_mean", ascending=False)
            df_pi.to_csv(OUT_DIR / "permutation_importance_test20.csv", index=False)
        except Exception as e:
            (OUT_DIR / "pi_test20_errors.log").write_text(
                f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                encoding="utf-8"
            )

    # -------------------------
    # Diagnostics
    # -------------------------
    if DO_CV_VARIABILITY:
        try:
            cv_variability_train80(
                X_tr=X_train80,
                y_tr=y_train80,
                params=model_params,
                rk_n_splits=RK_N_SPLITS,
                rk_n_repeats=RK_N_REPEATS,
                seed=SEED,
                n_jobs_rf=N_JOBS_RF,
                nrmse_den_mode=NRMSE_DEN_MODE,
                out_dir=OUT_DIR,
            )
        except Exception as e:
            (OUT_DIR / "cv_variability_errors.log").write_text(
                f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                encoding="utf-8",
            )

    if DO_LEARNING_CURVES_CV:
        try:
            run_learning_curves_cv_train80(
                X_train80=X_train80,
                y_train80=y_train80,
                model_params=model_params,
                out_dir=OUT_DIR,
                seed=SEED,
                fractions=list(LC_CV_FRACTIONS),
                n_splits=int(LC_CV_N_SPLITS),
                max_points=int(LC_CV_MAX_POINTS),
                min_points=int(LC_CV_MIN_POINTS),
                n_jobs_rf=N_JOBS_RF,
            )
        except Exception as e:
            (OUT_DIR / "learning_curve_cv_errors.log").write_text(
                f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                encoding="utf-8",
            )

    if DO_LEARNING_CURVE_HOLDOUT:
        try:
            learning_curve_holdout_r2_rmse(
                X_train80=X_train80,
                y_train80=y_train80,
                X_test20=X_test20,
                y_test20=y_test20,
                model_params=model_params,
                seed=SEED,
                train_sizes=list(LC_HOLDOUT_TRAIN_SIZES),
                repeats=int(LC_HOLDOUT_REPEATS),
                n_jobs_rf=N_JOBS_RF,
                out_dir=OUT_DIR,
            )
        except Exception as e:
            (OUT_DIR / "learning_curve_holdout_errors.log").write_text(
                f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                encoding="utf-8",
            )

    if DO_RESIDUAL_DIAGNOSTICS:
        try:
            run_residual_diagnostics_test20(
                ids_test20=ids_test20,
                y_true=y_test20,
                y_pred=yhat_test,
                out_dir=OUT_DIR,
                id_col=ID_COL,
            )
        except Exception as e:
            (OUT_DIR / "residual_diagnostics_errors.log").write_text(
                f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                encoding="utf-8",
            )

    if DO_OOB_DIAGNOSTICS:
        try:
            run_oob_diagnostics_train80(
                X_train80=X_train80,
                y_train80=y_train80,
                model_params=model_params,
                out_dir=OUT_DIR,
                seed=SEED,
                n_jobs_rf=N_JOBS_RF,
            )
        except Exception as e:
            (OUT_DIR / "oob_diagnostics_errors.log").write_text(
                f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                encoding="utf-8",
            )

    # -------------------------
    # Run summary
    # -------------------------
    save_json(
        {
            "TARGET": TARGET,
            "features_used": FEATURE_COLS,
            "excluded_interstitials": EXCLUDED_INTERSTITIALS,
            "n_total_used": int(len(y_all)),
            "n_train80": int(len(y_train80)),
            "n_test20": int(len(y_test20)),
            "metrics_train80": m_train,
            "metrics_test20": m_test,
            "selection_metric": "mean CV RMSE on train80",
            "model_file": "rf_model.joblib",
            "artifacts": {
                "clean_table": "modelling_table_clean.parquet",
                "train80": "train80_with_ids.parquet",
                "test20": "test20_with_ids.parquet",
                "pred_train80": "pred_train80_with_ids.parquet",
                "pred_test20": "pred_test20_with_ids.parquet",
                "best_params": "best_params_train80.json",
                "params_used": "model_params_used.json",
                "metrics_train80": "metrics_train80.csv",
                "metrics_test20": "metrics_test20.csv",
                "feature_importance_gini": "feature_importance_gini.csv",
                "permutation_importance_test20": "permutation_importance_test20.csv",
                "cv_scores": "cv_train80_scores.csv",
                "cv_summary": "cv_train80_summary.json",
                "learning_curve_cv_csv": "learning_curve_cv_train80_rmse.csv",
                "learning_curve_cv_png": "learning_curve_cv_train80_rmse.png",
                "learning_curve_holdout_raw": "learning_curve_holdout_raw.csv",
                "learning_curve_holdout_summary": "learning_curve_holdout_summary.csv",
                "learning_curve_holdout_r2_png": "learning_curve_holdout_r2.png",
                "learning_curve_holdout_rmse_png": "learning_curve_holdout_rmse.png",
                "residuals_test20": "residuals_test20.parquet",
                "resid_vs_yhat_test20_png": "resid_vs_yhat_test20.png",
                "hist_residuals_test20_png": "hist_residuals_test20.png",
                "oob_diagnostics": "oob_diagnostics.json",
            },
        },
        OUT_DIR / "run_summary.json",
    )

    print("OK:", OUT_DIR.resolve())

    # explicit cleanup
    del model
    gc.collect()


if __name__ == "__main__":
    main()