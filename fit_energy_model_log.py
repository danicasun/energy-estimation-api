#!/usr/bin/env python3
"""
Fit a log-transform energy prediction model from historical Slurm job data.

This model trains on log(energy + epsilon) instead of energy directly, which:
- Handles highly skewed energy distribution better
- Naturally prevents negative predictions
- Reduces impact of outliers
- Typically improves R² significantly

Historical CSV must contain columns:
User | Account | State | AllocTRES | ConsumedEnergyRaw | NCPUS | ReqMem | Submit | Start | End
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

from energy_constants import POWER_USAGE_EFFECTIVENESS
from utils import parse_memory, convert_energy_to_kwh


# -------------------------
# Data loading
# -------------------------

def load_historical_jobs(path) -> pd.DataFrame:
    """
    Load historical Slurm job data from either:
      - a single CSV file, or
      - a directory containing multiple CSVs.

    In this project, we expect the directory:
        "slurm_march_to_october data"
    inside the project root, containing several CSVs.
    """
    p = Path(path)
    if p.is_dir():
        csv_files = sorted(p.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {p}")
        # Historical Slurm exports are pipe-delimited ("|"), so specify the separator.
        dfs = [pd.read_csv(f, sep="|") for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(p, sep="|")
    return df


# -------------------------
# Model fitting
# -------------------------

def fit_energy_model_log(csv_path: str, model_out_path: str = "energy_model_log.json", 
                        epsilon: float = 0.001, min_energy: float = 0.0):
    """
    Fit energy prediction model using log-transform.
    
    Args:
        csv_path: Path to CSV file or directory of CSVs
        model_out_path: Where to save the model JSON
        epsilon: Small constant added before log-transform (default: 0.001 kWh)
        min_energy: Minimum energy threshold for training (default: 0.0, all jobs)
    """
    df = load_historical_jobs(csv_path)

    # ----- Keep only completed jobs -----
    df = df[df["State"] == "COMPLETED"].copy()

    # ----- Compute runtime -----
    df["Start"] = pd.to_datetime(df["Start"])
    df["End"] = pd.to_datetime(df["End"])
    df["runtime_hours"] = (df["End"] - df["Start"]).dt.total_seconds() / 3600

    # ----- Compute memory (GB) -----
    df["mem_gb_per_cpu"] = df["ReqMem"].apply(parse_memory)
    df["mem_gb_total"] = df["mem_gb_per_cpu"] * df["NCPUS"]

    # ----- Convert energy target to facility kWh using PUE -----
    df["energy_kwh"] = (
        df["ConsumedEnergyRaw"].apply(convert_energy_to_kwh) * POWER_USAGE_EFFECTIVENESS
    )

    # ----- Filter by minimum energy threshold -----
    if min_energy > 0:
        n_before = len(df)
        df = df[df["energy_kwh"] >= min_energy].copy()
        print(f"Filtered out {n_before - len(df)} jobs below {min_energy} kWh threshold")

    # ----- Drop unusable rows -----
    df = df.dropna(subset=["NCPUS", "mem_gb_total", "runtime_hours", "energy_kwh"])
    
    # ----- Ensure energy is non-negative -----
    df = df[df["energy_kwh"] >= 0].copy()

    # ----- Log-transform the target variable -----
    # Add epsilon to avoid log(0), then take log
    y_energy = df["energy_kwh"].values
    y_log = np.log(y_energy + epsilon)

    # ----- Feature matrix X -----
    # Columns used: intercept, CPUs, memory, runtime
    X = np.column_stack([
        np.ones(len(df)),
        df["NCPUS"].values,
        df["mem_gb_total"].values,
        df["runtime_hours"].values,
    ])

    # ----- Solve for beta via least squares on log-space -----
    beta, residuals, rank, s = np.linalg.lstsq(X, y_log, rcond=None)

    # ----- Estimate residual variance in log-space -----
    y_log_hat = X @ beta
    eps_log = y_log - y_log_hat
    n, p = X.shape
    sigma2_log = float(np.sum(eps_log**2) / max(n - p, 1))  # variance in log-space
    sigma_log = np.sqrt(sigma2_log)

    print("Log-transform fit complete.")
    print(f"beta (log-space) = {beta}")
    print(f"sigma (log-space) = {sigma_log:.4f}")
    print(f"Training samples: {n}")
    print(f"Epsilon (offset) = {epsilon}")

    # ----- Evaluate back-transformed predictions for comparison -----
    y_pred_energy = np.exp(y_log_hat) - epsilon
    mae = np.mean(np.abs(y_pred_energy - y_energy))
    rmse = np.sqrt(np.mean((y_pred_energy - y_energy)**2))
    print(f"Back-transformed MAE: {mae:.4f} kWh")
    print(f"Back-transformed RMSE: {rmse:.4f} kWh")

    # ----- Save model with log-transform metadata -----
    model = {
        "model_type": "log_transform",
        "beta": beta.tolist(),
        "sigma2_log": sigma2_log,
        "sigma_log": sigma_log,
        "epsilon": epsilon,
        "feature_order": ["intercept", "cpus", "mem_gb_total", "runtime_hours"],
        "n_training_samples": int(n),
        "min_energy_threshold": min_energy,
    }
    Path(model_out_path).write_text(json.dumps(model, indent=2))
    print(f"Saved log-transform model to {model_out_path}")


# -------------------------
# Entry point
# -------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Fit log-transform energy prediction model")
    ap.add_argument("--data", default=None, help="Path to CSV file or directory (default: 'slurm_march_to_october data')")
    ap.add_argument("--output", default="energy_model_log.json", help="Output model file path")
    ap.add_argument("--epsilon", type=float, default=0.001, help="Small constant for log-transform (default: 0.001 kWh)")
    ap.add_argument("--min-energy", type=float, default=0.0, help="Minimum energy threshold for training (default: 0.0)")
    args = ap.parse_args()
    
    # Default to the same directory used in fit_energy_model.py
    if args.data is None:
        project_root = Path(__file__).resolve().parent
        historical_dir = project_root / "slurm_march_to_october data"
        csv_path = historical_dir
    else:
        csv_path = args.data
    
    fit_energy_model_log(csv_path, args.output, args.epsilon, args.min_energy)

