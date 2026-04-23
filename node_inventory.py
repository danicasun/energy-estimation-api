"""Node inventory lookup utilities for Sherlock machine metadata."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional


INVENTORY_FILENAME = "sherlock_all_machines.csv"


@dataclass(frozen=True)
class NodeHardwareProfile:
    """Hardware attributes for a single node from inventory."""

    node_name: str
    cpu_core_count: int
    gpu_count: int
    cpu_generation: Optional[str] = None
    cpu_sku: Optional[str] = None
    cpu_manufacturer: Optional[str] = None
    node_class: Optional[str] = None


@dataclass(frozen=True)
class NodeInventoryAggregation:
    """Aggregated hardware counts and lookup diagnostics."""

    total_cpu_cores: int
    total_gpu_count: int
    matched_node_names: List[str]
    missing_node_names: List[str]


def _default_inventory_path() -> Path:
    return Path(__file__).resolve().parent / INVENTORY_FILENAME


@lru_cache(maxsize=1)
def load_node_inventory() -> Dict[str, NodeHardwareProfile]:
    """Load node inventory from sherlock CSV file."""
    inventory_path = _default_inventory_path()
    return load_node_inventory_from_path(inventory_path)


def load_node_inventory_from_path(inventory_path: Path) -> Dict[str, NodeHardwareProfile]:
    """Load node inventory from a CSV path."""
    if not inventory_path.exists():
        return {}

    inventory_by_node_name: Dict[str, NodeHardwareProfile] = {}
    with inventory_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            raw_node_name = (row.get("node") or "").strip()
            if not raw_node_name:
                continue

            node_name = raw_node_name.lower()
            cpu_core_count = _safe_int(row.get("num_cpus"), fallback=0)
            gpu_count = _safe_int(row.get("num_gpus"), fallback=0)

            inventory_by_node_name[node_name] = NodeHardwareProfile(
                node_name=node_name,
                cpu_core_count=cpu_core_count,
                gpu_count=gpu_count,
                cpu_generation=_clean_optional_text(row.get("CPU_GEN")),
                cpu_sku=_clean_optional_text(row.get("CPU_SKU")),
                cpu_manufacturer=_clean_optional_text(row.get("CPU_MNF")),
                node_class=_clean_optional_text(row.get("CLASS")),
            )

    return inventory_by_node_name


def lookup_node_profile(node_name: str) -> Optional[NodeHardwareProfile]:
    """Lookup a node hardware profile by exact node name."""
    normalized_node_name = node_name.strip().lower()
    if not normalized_node_name:
        return None
    inventory = load_node_inventory()
    return inventory.get(normalized_node_name)


def aggregate_allocated_nodes(node_names: List[str]) -> NodeInventoryAggregation:
    """Aggregate CPU/GPU counts for allocated nodes from inventory."""
    total_cpu_cores = 0
    total_gpu_count = 0
    matched_node_names: List[str] = []
    missing_node_names: List[str] = []

    for node_name in node_names:
        profile = lookup_node_profile(node_name)
        if profile is None:
            missing_node_names.append(node_name)
            continue
        matched_node_names.append(profile.node_name)
        total_cpu_cores += max(profile.cpu_core_count, 0)
        total_gpu_count += max(profile.gpu_count, 0)

    return NodeInventoryAggregation(
        total_cpu_cores=total_cpu_cores,
        total_gpu_count=total_gpu_count,
        matched_node_names=matched_node_names,
        missing_node_names=missing_node_names,
    )


def _safe_int(raw_value: Optional[str], fallback: int) -> int:
    try:
        return int((raw_value or "").strip())
    except ValueError:
        return fallback


def _clean_optional_text(raw_value: Optional[str]) -> Optional[str]:
    cleaned = (raw_value or "").strip()
    return cleaned or None
