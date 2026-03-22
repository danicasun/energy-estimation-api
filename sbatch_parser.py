"""Parse SBATCH directives into structured parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SbatchParameters:
    """Structured SBATCH parameters with explicit units."""

    partition_name: Optional[str] = None
    node_count: Optional[int] = None
    cpu_cores_per_task: Optional[int] = None
    task_count: Optional[int] = None
    gpu_count: Optional[int] = None
    memory_gigabytes_total: Optional[float] = None
    memory_gigabytes_per_cpu: Optional[float] = None
    walltime_hours: Optional[float] = None
    nodelist: Optional[str] = None


def parse_walltime_hours(raw_value: str) -> Optional[float]:
    """Parse SLURM time strings to hours (hours units)."""
    trimmed_value = raw_value.strip()
    if not trimmed_value:
        return None

    day_split = trimmed_value.split("-")
    days = 0
    time_part = trimmed_value
    if len(day_split) == 2:
        try:
            days = int(day_split[0])
        except ValueError:
            return None
        time_part = day_split[1]

    time_segments = time_part.split(":")
    if any(segment.strip() == "" for segment in time_segments):
        return None

    try:
        numeric_segments = [int(segment) for segment in time_segments]
    except ValueError:
        return None

    hours = 0
    minutes = 0
    seconds = 0
    if len(numeric_segments) == 3:
        hours, minutes, seconds = numeric_segments
    elif len(numeric_segments) == 2:
        hours, minutes = numeric_segments
    elif len(numeric_segments) == 1:
        hours = numeric_segments[0]
    else:
        return None

    return days * 24 + hours + minutes / 60 + seconds / 3600


def parse_memory_to_gigabytes(raw_value: str) -> Optional[float]:
    """Parse memory strings (e.g., 32G, 1024M) into GB units."""
    trimmed_value = raw_value.strip().upper()
    if not trimmed_value:
        return None

    value_part = ""
    unit_part = ""
    for char in trimmed_value:
        if char.isdigit() or char == ".":
            value_part += char
        else:
            unit_part += char

    try:
        numeric_value = float(value_part)
    except ValueError:
        return None

    unit = unit_part.strip() or "G"
    if unit in {"K", "KB"}:
        return numeric_value / (1024 * 1024)
    if unit in {"M", "MB"}:
        return numeric_value / 1024
    if unit in {"G", "GB"}:
        return numeric_value
    if unit in {"T", "TB"}:
        return numeric_value * 1024
    if unit in {"P", "PB"}:
        return numeric_value * 1024 * 1024
    return None


def parse_sbatch_text(sbatch_text: str) -> SbatchParameters:
    """Parse SBATCH text into structured parameters."""
    parameters = SbatchParameters()
    directives = [
        line.strip()
        for line in sbatch_text.splitlines()
        if line.strip().startswith("#SBATCH")
    ]

    for directive in directives:
        normalized = directive.replace("#SBATCH", "", 1).strip()

        if normalized.startswith("--partition"):
            value = _split_sbatch_value(normalized)
            parameters.partition_name = value or parameters.partition_name

        if normalized.startswith("--nodes"):
            value = _split_sbatch_value(normalized)
            parameters.node_count = _parse_int(value, parameters.node_count)

        if normalized.startswith("--ntasks"):
            value = _split_sbatch_value(normalized)
            parameters.task_count = _parse_int(value, parameters.task_count)

        if normalized.startswith("--cpus-per-task"):
            value = _split_sbatch_value(normalized)
            parameters.cpu_cores_per_task = _parse_int(value, parameters.cpu_cores_per_task)

        if normalized.startswith("--gres"):
            gpu_count = _parse_gres_gpu_count(normalized)
            if gpu_count is not None:
                parameters.gpu_count = gpu_count

        if normalized.startswith("--mem-per-cpu"):
            value = _split_sbatch_value(normalized)
            mem_per_cpu = parse_memory_to_gigabytes(value) if value else None
            if mem_per_cpu is not None:
                parameters.memory_gigabytes_per_cpu = mem_per_cpu

        if normalized.startswith("--mem"):
            value = _split_sbatch_value(normalized)
            mem_total = parse_memory_to_gigabytes(value) if value else None
            if mem_total is not None:
                parameters.memory_gigabytes_total = mem_total

        if normalized.startswith("--time"):
            value = _split_sbatch_value(normalized)
            walltime_hours = parse_walltime_hours(value) if value else None
            if walltime_hours is not None:
                parameters.walltime_hours = walltime_hours

        if normalized.startswith("--nodelist"):
            value = _split_sbatch_value(normalized)
            parameters.nodelist = value or parameters.nodelist

    return parameters


def _split_sbatch_value(normalized_directive: str) -> Optional[str]:
    parts = normalized_directive.split("=", 1)
    if len(parts) == 2:
        return parts[1].strip()

    spaced_parts = normalized_directive.split()
    if len(spaced_parts) >= 2:
        return spaced_parts[1].strip()

    return None


def _parse_int(raw_value: Optional[str], fallback: Optional[int]) -> Optional[int]:
    if raw_value is None:
        return fallback
    try:
        return int(raw_value)
    except ValueError:
        return fallback


def _parse_gres_gpu_count(normalized_directive: str) -> Optional[int]:
    # Examples: --gres=gpu:4, --gres=gpu:V100:4
    directive = normalized_directive.replace("--gres", "", 1).strip()
    parts = directive.split("=", 1)
    gres_value = parts[1] if len(parts) == 2 else (parts[0] if parts else "")
    if "gpu" not in gres_value:
        return None

    segments = gres_value.split(":")
    for segment in reversed(segments):
        if segment.isdigit():
            return int(segment)
    return None
