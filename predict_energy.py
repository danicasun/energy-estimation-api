#!/usr/bin/env python3
"""
Unified energy prediction script from SBATCH file.

Supports:
- Both analytical (fast) and Monte Carlo (full distribution) methods
- Both standard and log-transform models
- Automatic model type detection

Requires:
    - energy_model.json or energy_model_log.json (from fit_energy_model.py or fit_energy_model_log.py)
    - an SBATCH file as input
"""

import json
from typing import Dict, Literal
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from energy_constants import POWER_USAGE_EFFECTIVENESS
from utils import parse_sbatch_file, normalize_memory, parse_time_limit, load_model


def predict_energy(
    sbatch_path: str,
    model_path: str = "energy_model.json",
    method: Literal["analytical", "monte_carlo"] = "analytical",
    n_samples: int = 2000,
    clamp_negative: bool = True,
    user: str = None
) -> Dict:
    """
    Predict energy usage from SBATCH file.
    
    Args:
        sbatch_path: Path to SBATCH file
        model_path: Path to model JSON file
        method: "analytical" for fast Normal percentiles, "monte_carlo" for full distribution
        n_samples: Number of samples for Monte Carlo method
        clamp_negative: If True, clamp negative predictions to 0
        user: User ID for user-specific models (required if model is user-specific)
    
    Returns:
        Dictionary with prediction results
    """
    # Load model
    all_models = load_model(model_path)
    model_type = all_models.get("model_type", "standard")
    
    # Handle user-specific models
    if model_type == "user_specific":
        if user is None:
            raise ValueError("User-specific model requires --user parameter")
        
        # Try to get user-specific model, fallback to global
        user_models = all_models.get("user_models", {})
        if user in user_models:
            model = user_models[user]
            model_source = f"user-specific ({user})"
        else:
            model = all_models["global_model"]
            model_source = "global (fallback)"
        
        # User-specific models are always standard type
        model_type = "standard"
    else:
        # Standard or log-transform model
        model = all_models
        model_source = model_type
    
    # Extract model parameters
    if model_type == "log_transform":
        beta = np.array(model["beta"])
        sigma_log = float(model.get("sigma_log", model.get("sigma", 1.0)))
        epsilon = float(model.get("epsilon", 0.001))
        sigma = None
    else:
        beta = np.array(model["beta"])
        sigma = float(model.get("sigma", 1.0))
        sigma_log = None
        epsilon = None
    
    # Extract features from SBATCH
    sb = parse_sbatch_file(sbatch_path)
    cpus = int(sb.get("cpus-per-task") or 1) * int(sb.get("ntasks") or 1)
    mem_per_cpu_gb = normalize_memory(sb.get("mem-per-cpu") or "0G")
    mem_gb_total = mem_per_cpu_gb * cpus
    requested_hours = parse_time_limit(sb.get("time") or "0:00:00")
    
    # Feature vector
    x = np.array([1.0, cpus, mem_gb_total, requested_hours])
    
    # Predict in model's native space (linear or log)
    mu_native = float(x @ beta)
    
    if model_type == "log_transform":
        # For log-transform: mu_native is in log-space
        # Convert to energy space: exp(mu_log) - epsilon
        mu_energy = np.exp(mu_native) - epsilon
        std_energy = None  # Will compute from samples for log-transform
    else:
        # Standard model: already in energy space
        mu_energy = mu_native
        std_energy = sigma

    # Convert model output to facility-level energy using PUE.
    mu_energy *= POWER_USAGE_EFFECTIVENESS
    if std_energy is not None:
        std_energy *= POWER_USAGE_EFFECTIVENESS
    
    result = {
        "cpus": cpus,
        "mem_gb_total": mem_gb_total,
        "requested_hours": requested_hours,
        "model_type": model_type,
        "method": method,
        "pue": POWER_USAGE_EFFECTIVENESS,
    }
    
    # Add user info if using user-specific models
    if all_models.get("model_type") == "user_specific":
        result["user"] = user
        result["model_source"] = model_source
    
    if method == "analytical":
        # Fast analytical method using Normal distribution
        if model_type == "log_transform":
            # For log-transform, we need to sample to get proper distribution
            # Approximate using log-normal properties
            # If Y = exp(X) where X ~ N(mu, sigma^2), then E[Y] = exp(mu + sigma^2/2)
            mu_energy = (np.exp(mu_native + sigma_log**2 / 2) - epsilon) * POWER_USAGE_EFFECTIVENESS
            
            # For log-normal: Var[Y] = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
            var_energy = (np.exp(sigma_log**2) - 1) * np.exp(2 * mu_native + sigma_log**2)
            var_energy *= POWER_USAGE_EFFECTIVENESS ** 2
            std_energy = np.sqrt(var_energy)
            
            # Percentiles: sample to get accurate percentiles for log-normal
            samples_log = np.random.normal(loc=mu_native, scale=sigma_log, size=10000)
            samples_energy = (
                np.maximum(np.exp(samples_log) - epsilon, 0)
                if clamp_negative
                else np.exp(samples_log) - epsilon
            )
            samples_energy = samples_energy * POWER_USAGE_EFFECTIVENESS
            p10 = np.percentile(samples_energy, 10)
            p50 = np.percentile(samples_energy, 50)
            p90 = np.percentile(samples_energy, 90)
        else:
            # Standard model: use Normal distribution directly
            std_energy = sigma * POWER_USAGE_EFFECTIVENESS
            p10 = mu_energy + (-1.28155) * std_energy
            p50 = mu_energy
            p90 = mu_energy + (1.28155) * std_energy
            
            if clamp_negative:
                p10 = max(p10, 0.0)
                p50 = max(p50, 0.0)
                p90 = max(p90, 0.0)
        
        if clamp_negative:
            mu_energy = max(mu_energy, 0.0)
        
        result.update({
            "mean_energy_kwh": float(mu_energy),
            "std_energy_kwh": float(std_energy) if std_energy is not None else None,
            "p10_energy_kwh": float(p10),
            "p50_energy_kwh": float(p50),
            "p90_energy_kwh": float(p90),
        })
    
    else:  # monte_carlo
        # Full Monte Carlo sampling
        if model_type == "log_transform":
            # Sample in log-space, then transform
            samples_log = np.random.normal(loc=mu_native, scale=sigma_log, size=n_samples)
            samples_energy = (np.exp(samples_log) - epsilon) * POWER_USAGE_EFFECTIVENESS
        else:
            # Sample directly in energy space
            samples_energy = np.random.normal(
                loc=mu_energy,
                scale=sigma * POWER_USAGE_EFFECTIVENESS,
                size=n_samples,
            )
        
        # Clamp negative values if requested
        if clamp_negative:
            samples_energy = np.maximum(samples_energy, 0.0)
        
        result.update({
            "mean_energy_kwh": float(np.mean(samples_energy)),
            "std_energy_kwh": float(np.std(samples_energy)),
            "min_energy_kwh": float(np.min(samples_energy)),
            "max_energy_kwh": float(np.max(samples_energy)),
            "p10_energy_kwh": float(np.percentile(samples_energy, 10)),
            "p50_energy_kwh": float(np.percentile(samples_energy, 50)),
            "p90_energy_kwh": float(np.percentile(samples_energy, 90)),
            "n_samples": n_samples,
        })
    
    return result


def plot_energy_distribution(
    mean: float,
    std: float,
    model_type: str = "standard",
    samples: np.ndarray = None
) -> None:
    """
    Visualize the energy prediction distribution.
    
    Args:
        mean: Mean energy prediction (kWh)
        std: Standard deviation (kWh)
        model_type: "standard" or "log_transform"
        samples: Optional array of samples for Monte Carlo method
    """
    if samples is not None:
        # Plot histogram of samples
        plt.figure(figsize=(8, 5))
        plt.hist(samples, bins=50, density=True, alpha=0.6, label="MC Samples")
        plt.axvline(mean, color="red", linestyle="--", linewidth=2, label="Mean")
        plt.xlabel("Energy (kWh)")
        plt.ylabel("Density")
        plt.title("Monte-Carlo Energy Usage Distribution")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
    else:
        # Plot analytical Normal distribution
        if std <= 0:
            plt.figure(figsize=(8, 5))
            plt.axvline(mean, color="red", linestyle="--", linewidth=2, label="Mean")
            plt.xlabel("Energy (kWh)")
            plt.ylabel("Probability Density")
            plt.title("Predicted Energy Usage (Deterministic)")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.show()
            return
        
        plt.figure(figsize=(8, 5))
        xs = np.linspace(max(0, mean - 4 * std), mean + 4 * std, 300)
        ys = norm.pdf(xs, loc=mean, scale=std)
        plt.plot(xs, ys, label="Energy Distribution (Normal)")
        plt.axvline(mean, color="red", linestyle="--", linewidth=2, label="Mean")
        plt.xlabel("Energy (kWh)")
        plt.ylabel("Probability Density")
        plt.title("Predicted Energy Usage Distribution")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()


# -------------------------
# CLI
# -------------------------

if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser(
        description="Predict energy usage (kWh) from SBATCH file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick analytical prediction (default)
  python predict_energy.py job.sbatch
  
  # Use user-specific model
  python predict_energy.py job.sbatch --model energy_model_by_user.json --user rachelq
  
  # Use Monte Carlo method with custom samples
  python predict_energy.py job.sbatch --method monte_carlo --samples 5000
  
  # Use log-transform model
  python predict_energy.py job.sbatch --model energy_model_log.json
  
  # Show distribution plot
  python predict_energy.py job.sbatch --plot
        """
    )
    ap.add_argument("SBATCH_FILE", help="Path to an SBATCH script")
    ap.add_argument("--model", default="energy_model.json", help="Path to trained model JSON")
    ap.add_argument(
        "--method",
        choices=["analytical", "monte_carlo"],
        default="analytical",
        help="Prediction method: 'analytical' (fast) or 'monte_carlo' (full distribution)"
    )
    ap.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Number of samples for Monte Carlo method (default: 2000)"
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Show distribution plot of predicted energy"
    )
    ap.add_argument(
        "--no-clamp",
        action="store_true",
        help="Don't clamp negative predictions to zero"
    )
    ap.add_argument(
        "--user",
        default=None,
        help="User ID for user-specific models (required if using user-specific model)"
    )
    args = ap.parse_args()
    
    # Run prediction
    result = predict_energy(
        sbatch_path=args.SBATCH_FILE,
        model_path=args.model,
        method=args.method,
        n_samples=args.samples,
        clamp_negative=not args.no_clamp,
        user=args.user
    )
    
    # Print results (remove samples if present to avoid huge output)
    output = {k: v for k, v in result.items() if k != "samples"}
    print(json.dumps(output, indent=2))
    
    # Plot if requested
    if args.plot:
        if args.method == "monte_carlo":
            # For MC, we'd need to regenerate samples for plotting
            # Re-run to get samples for visualization
            result_with_samples = predict_energy(
                sbatch_path=args.SBATCH_FILE,
                model_path=args.model,
                method="monte_carlo",
                n_samples=args.samples,
                clamp_negative=not args.no_clamp
            )
            # Generate samples for plotting
            model = load_model(args.model)
            model_type = model.get("model_type", "standard")
            beta = np.array(model["beta"])
            
            sb = parse_sbatch_file(args.SBATCH_FILE)
            cpus = int(sb.get("cpus-per-task") or 1) * int(sb.get("ntasks") or 1)
            mem_per_cpu_gb = normalize_memory(sb.get("mem-per-cpu") or "0G")
            mem_gb_total = mem_per_cpu_gb * cpus
            requested_hours = parse_time_limit(sb.get("time") or "0:00:00")
            x = np.array([1.0, cpus, mem_gb_total, requested_hours])
            
            mu_native = float(x @ beta)
            if model_type == "log_transform":
                sigma_log = float(model.get("sigma_log", 1.0))
                epsilon = float(model.get("epsilon", 0.001))
                samples_log = np.random.normal(loc=mu_native, scale=sigma_log, size=args.samples)
                samples = np.exp(samples_log) - epsilon
            else:
                sigma = float(model.get("sigma", 1.0))
                samples = np.random.normal(loc=mu_native, scale=sigma, size=args.samples)
            
            if not args.no_clamp:
                samples = np.maximum(samples, 0.0)
            
            plot_energy_distribution(
                mean=result["mean_energy_kwh"],
                std=result["std_energy_kwh"],
                model_type=model_type,
                samples=samples
            )
        else:
            plot_energy_distribution(
                mean=result["mean_energy_kwh"],
                std=result.get("std_energy_kwh", 0),
                model_type=result["model_type"]
            )

