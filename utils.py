#!/usr/bin/env python3
"""
Shared utility functions for the energy prediction scripts.
"""

from pathlib import Path
from typing import Dict
import json
import re
import numpy as np
import pandas as pd


# -------------------------
# Memory parsing
# -------------------------

def normalize_memory(mem_str: str) -> float:
    """
    Convert a Slurm-style memory string to GB.

    Examples:
        '4G', '4096M', '8GB', '16384MB'
    """
    if mem_str is None or (isinstance(mem_str, float) and np.isnan(mem_str)):
        return np.nan

    mem_str = str(mem_str).upper().strip()
    if mem_str == "":
        return np.nan

    num = "".join(ch for ch in mem_str if ch.isdigit() or ch == ".")
    unit = "".join(ch for ch in mem_str if ch.isalpha())

    if num == "":
        return np.nan

    val = float(num)

    if "G" in unit:
        return val
    if "M" in unit:
        return val / 1024
    if "K" in unit:  # almost never appears
        return val / (1024 * 1024)

    # default to MB if no/unknown unit
    return val / 1024


# Backwards-compatible alias for scripts that used parse_memory
parse_memory = normalize_memory


# -------------------------
# Energy units
# -------------------------

def convert_energy_to_kwh(raw):
    """
    Convert raw energy to kWh.

    Assumes raw is in Joules (Slurm's ConsumedEnergyRaw), so:
        kWh = J / 3.6e6
    """
    return raw / 3.6e6


# -------------------------
# SBATCH parsing
# -------------------------

def parse_sbatch_file(filepath: str) -> Dict[str, str]:
    """
    Parse key SLURM directives from an SBATCH file.

    Returns a dict with keys:
        'cpus-per-task', 'ntasks', 'mem-per-cpu', 'time'
    """
    text = Path(filepath).read_text()
    parsed: Dict[str, str] = {
        "cpus-per-task": None,
        "ntasks": None,
        "mem-per-cpu": None,
        "time": None,
    }
    pat = re.compile(r"#SBATCH\s+--?([\w-]+)\s*(?:=\s*)?(\S+)")

    for line in text.splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        key, val = m.groups()
        if key in parsed:
            parsed[key] = val

    return parsed


def parse_time_limit(time_str: str) -> float:
    """
    Convert a Slurm --time value into hours.

    Supports:
        HH:MM:SS
        HH:MM
        DD-HH:MM:SS
    """
    if time_str is None:
        return np.nan

    time_str = time_str.strip()
    if time_str == "":
        return np.nan

    if "-" in time_str:
        d, rest = time_str.split("-", 1)
        days = int(d)
    else:
        days = 0
        rest = time_str

    parts = rest.split(":")
    if len(parts) == 3:
        h, m, s = map(int, parts)
    elif len(parts) == 2:
        h, m = map(int, parts)
        s = 0
    else:
        return np.nan

    return days * 24 + h + m / 60 + s / 3600


# -------------------------
# Model loading
# -------------------------

def load_model(path: str = "models/old_models/energy_model.json") -> Dict:
    """Load the fitted energy model JSON file."""
    return json.loads(Path(path).read_text())


def load_energy_model(path: str = "models/old_models/energy_model.json") -> Dict:
    """
    Backwards-compatible alias used by some scripts.
    """
    return load_model(path)


# -------------------------
# Historical job loading
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
