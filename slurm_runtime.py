"""Read Slurm runtime environment variables and scontrol metadata."""

import os
import subprocess
from typing import Dict, List, Optional


def get_slurm_env_vars() -> Dict[str, Optional[str]]:
    """
    Read Slurm environment variables for a running job.
    
    Returns:
        Dictionary with Slurm environment variables:
        - SLURM_CPUS_PER_TASK
        - SLURM_NTASKS
        - SLURM_MEM_PER_CPU (if available)
        - SLURM_TIME_LIMIT
        - SLURM_JOB_NODELIST
        - SLURM_JOB_ID
    """
    env_vars = {
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "ntasks": os.environ.get("SLURM_NTASKS"),
        "mem_per_cpu": os.environ.get("SLURM_MEM_PER_CPU"),
        "time": os.environ.get("SLURM_TIME_LIMIT"),
        "nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }
    
    return env_vars


def get_job_runtime(job_id: Optional[str] = None) -> Optional[float]:
    """
    Get actual runtime of a job in hours using scontrol.
    
    Args:
        job_id: Job ID (defaults to SLURM_JOB_ID from environment)
        
    Returns:
        Runtime in hours (float) or None if unavailable
    """
    if not job_id:
        job_id = os.environ.get("SLURM_JOB_ID")
    
    if not job_id:
        return None
    
    try:
        # Use scontrol to get job information
        result = subprocess.run(
            ["scontrol", "show", "job", str(job_id)],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            return None
        
        # Parse RunTime from output
        # Format: RunTime=00:15:30 or RunTime=1-12:30:00
        for line in result.stdout.split('\n'):
            if 'RunTime=' in line:
                runtime_str = line.split('RunTime=')[1].split()[0]
                return parse_runtime_string(runtime_str)
        
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        return None


def parse_runtime_string(runtime_str: str) -> float:
    """
    Parse Slurm runtime string to hours.
    
    Formats: "HH:MM:SS" or "DD-HH:MM:SS"
    
    Args:
        runtime_str: Runtime string from scontrol
        
    Returns:
        Time in hours (float)
    """
    if not runtime_str or runtime_str == "UNLIMITED":
        return 0.0
    
    # Check for day-hour format
    if '-' in runtime_str:
        parts = runtime_str.split('-')
        days = int(parts[0])
        time_part = parts[1]
    else:
        days = 0
        time_part = runtime_str
    
    # Parse time part: HH:MM:SS
    time_parts = time_part.split(':')
    if len(time_parts) == 3:
        hours = int(time_parts[0])
        minutes = int(time_parts[1])
        seconds = int(time_parts[2])
    else:
        raise ValueError(f"Cannot parse runtime format: {runtime_str}")
    
    total_hours = days * 24 + hours + minutes / 60.0 + seconds / 3600.0
    return total_hours


def get_node_prefix(nodelist: Optional[str] = None) -> Optional[str]:
    """
    Extract node prefix from SLURM_JOB_NODELIST.
    
    Examples:
        "sh-01-01" -> "sh"
        "node001" -> "node"
        "sh-01-[01-10]" -> "sh"
    
    Args:
        nodelist: Node list string (defaults to SLURM_JOB_NODELIST)
        
    Returns:
        Node prefix string or None
    """
    if not nodelist:
        nodelist = os.environ.get("SLURM_JOB_NODELIST")
    
    if not nodelist:
        return None
    
    # Extract prefix (everything before the first digit or dash-number pattern)
    # Handle formats like "sh-01-01", "node001", "sh-01-[01-10]"
    match = nodelist.split('-')[0] if '-' in nodelist else nodelist.rstrip('0123456789[]')
    
    # Remove trailing digits
    prefix = ''.join(c for c in match if not c.isdigit())
    
    return prefix if prefix else None


def expand_slurm_nodelist(nodelist: Optional[str]) -> List[str]:
    """Expand a Slurm nodelist expression into concrete node names."""
    if not nodelist:
        return []

    trimmed_nodelist = nodelist.strip()
    if not trimmed_nodelist:
        return []

    resolved_nodes = _expand_with_scontrol(trimmed_nodelist)
    if resolved_nodes:
        return resolved_nodes

    return _expand_nodelist_fallback(trimmed_nodelist)


def _expand_with_scontrol(nodelist: str) -> List[str]:
    try:
        result = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []

    if result.returncode != 0:
        return []

    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _expand_nodelist_fallback(nodelist: str) -> List[str]:
    expanded_node_names: List[str] = []
    for token in _split_nodelist_tokens(nodelist):
        expanded_node_names.extend(_expand_nodelist_token(token))
    return expanded_node_names


def _split_nodelist_tokens(nodelist: str) -> List[str]:
    tokens: List[str] = []
    current_token: List[str] = []
    bracket_depth = 0

    for character in nodelist:
        if character == "," and bracket_depth == 0:
            token = "".join(current_token).strip()
            if token:
                tokens.append(token)
            current_token = []
            continue
        if character == "[":
            bracket_depth += 1
        elif character == "]":
            bracket_depth = max(0, bracket_depth - 1)
        current_token.append(character)

    trailing_token = "".join(current_token).strip()
    if trailing_token:
        tokens.append(trailing_token)
    return tokens


def _expand_nodelist_token(token: str) -> List[str]:
    if "[" not in token or "]" not in token:
        return [token]

    prefix, remainder = token.split("[", 1)
    bracket_content, suffix = remainder.split("]", 1)
    expanded_suffixes: List[str] = []

    for bracket_segment in bracket_content.split(","):
        trimmed_segment = bracket_segment.strip()
        if not trimmed_segment:
            continue
        if "-" in trimmed_segment:
            start_raw, end_raw = trimmed_segment.split("-", 1)
            if not start_raw.isdigit() or not end_raw.isdigit():
                continue
            width = max(len(start_raw), len(end_raw))
            start_value = int(start_raw)
            end_value = int(end_raw)
            if end_value < start_value:
                continue
            for value in range(start_value, end_value + 1):
                expanded_suffixes.append(str(value).zfill(width))
        else:
            expanded_suffixes.append(trimmed_segment)

    if not expanded_suffixes:
        return [token]

    return [f"{prefix}{expanded_suffix}{suffix}" for expanded_suffix in expanded_suffixes]

