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
    ntasks_per_node: Optional[int] = None


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
    has_day_prefix = len(day_split) == 2
    if has_day_prefix:
        # Slurm with day prefix allows D-HH, D-HH:MM, and D-HH:MM:SS.
        if len(numeric_segments) == 3:
            hours, minutes, seconds = numeric_segments
        elif len(numeric_segments) == 2:
            hours, minutes = numeric_segments
        elif len(numeric_segments) == 1:
            hours = numeric_segments[0]
        else:
            return None
    else:
        # Slurm without day prefix treats one/two segments as MM or MM:SS.
        if len(numeric_segments) == 3:
            hours, minutes, seconds = numeric_segments
        elif len(numeric_segments) == 2:
            minutes, seconds = numeric_segments
        elif len(numeric_segments) == 1:
            minutes = numeric_segments[0]
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

    unit = unit_part.strip() or "M"
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
        flag_name, value = _split_sbatch_flag_and_value(normalized)
        if flag_name is None:
            continue

        if flag_name in {"--partition", "-p"}:
            parameters.partition_name = value or parameters.partition_name

        elif flag_name in {"--nodes", "-N"}:
            parameters.node_count = _parse_int(value, parameters.node_count)

        elif flag_name == "--ntasks-per-node":
            parameters.ntasks_per_node = _parse_int(value, parameters.ntasks_per_node)

        elif flag_name in {"--ntasks", "-n"}:
            parameters.task_count = _parse_int(value, parameters.task_count)

        elif flag_name in {"--cpus-per-task", "-c"}:
            parameters.cpu_cores_per_task = _parse_int(value, parameters.cpu_cores_per_task)

        elif flag_name == "--gres":
            gpu_count = _parse_gpu_count_value(value)
            if gpu_count is not None:
                parameters.gpu_count = gpu_count

        elif flag_name in {"--gpus", "-G"}:
            gpu_count = _parse_gpu_count_value(value)
            if gpu_count is not None:
                parameters.gpu_count = gpu_count

        elif flag_name == "--mem-per-cpu":
            mem_per_cpu = parse_memory_to_gigabytes(value) if value else None
            if mem_per_cpu is not None:
                parameters.memory_gigabytes_per_cpu = mem_per_cpu

        elif flag_name == "--mem":
            mem_total = parse_memory_to_gigabytes(value) if value else None
            if mem_total is not None:
                parameters.memory_gigabytes_total = mem_total

        elif flag_name in {"--time", "-t"}:
            walltime_hours = parse_walltime_hours(value) if value else None
            if walltime_hours is not None:
                parameters.walltime_hours = walltime_hours

        elif flag_name in {"--nodelist", "-w"}:
            parameters.nodelist = value or parameters.nodelist

    return parameters


def _split_sbatch_flag_and_value(normalized_directive: str) -> tuple[Optional[str], Optional[str]]:
    """Split a directive into the exact flag name and its value."""
    trimmed_directive = normalized_directive.strip()
    if not trimmed_directive.startswith("-"):
        return None, None

    if trimmed_directive.startswith("--"):
        flag_body = trimmed_directive[2:]
        if not flag_body:
            return None, None
        if "=" in flag_body:
            flag_name, raw_value = flag_body.split("=", 1)
            stripped_value = raw_value.strip()
            return f"--{flag_name.strip()}", stripped_value or None
        spaced_parts = flag_body.split(None, 1)
        flag_name = spaced_parts[0]
        raw_value = spaced_parts[1].strip() if len(spaced_parts) > 1 else None
        return f"--{flag_name}", raw_value or None

    if len(trimmed_directive) < 2 or not trimmed_directive[1].isalpha():
        return None, None

    flag_name = f"-{trimmed_directive[1]}"
    remainder = trimmed_directive[2:]
    if remainder.startswith("="):
        remainder = remainder[1:]
    remainder = remainder.strip()
    return flag_name, remainder or None


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
    _, gres_value = _split_sbatch_flag_and_value(normalized_directive)
    return _parse_gpu_count_value(gres_value)


def _parse_gpu_count_value(raw_value: Optional[str]) -> Optional[int]:
    if raw_value is None:
        return None
    trimmed_value = raw_value.strip()
    if not trimmed_value:
        return None
    if trimmed_value.isdigit():
        return int(trimmed_value)
    if "gpu" not in trimmed_value:
        return None

    segments = trimmed_value.split(":")
    for segment in reversed(segments):
        if segment.isdigit():
            return int(segment)
    return None
