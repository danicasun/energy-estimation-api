"""Tests for node inventory loading and aggregation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from node_inventory import aggregate_allocated_nodes, load_node_inventory, load_node_inventory_from_path


class TestNodeInventory(unittest.TestCase):
    def test_load_node_inventory_from_path_reads_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_path = Path(temporary_directory) / "inventory.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "node,num_cpus,num_gpus,CPU_GEN,GPU_SKU,IB,CPU_FRQ,GPU_GEN,CLASS,CPU_SKU,CPU_MNF,GPU_MEM,GPU_CC,GPU_BRD",
                        "sh02-01n61,20,0,BGM,,NDR,2.25GHz,,SH4_CSCALE,9754,AMD,,,",
                        "sh03-01n01,52,4,ICL,A100,NDR,2.0GHz,Ampere,SH4_GPU,8358,Intel,80GB,8.0,NVIDIA",
                    ]
                ),
                encoding="utf-8",
            )

            inventory = load_node_inventory_from_path(csv_path)

        self.assertEqual(inventory["sh02-01n61"].cpu_core_count, 20)
        self.assertEqual(inventory["sh03-01n01"].gpu_count, 4)
        self.assertEqual(inventory["sh03-01n01"].cpu_manufacturer, "Intel")

    def test_aggregate_allocated_nodes_reports_missing(self) -> None:
        load_node_inventory.cache_clear()
        aggregation = aggregate_allocated_nodes(["sh02-01n61", "missing-node"])

        self.assertEqual(aggregation.total_cpu_cores, 20)
        self.assertEqual(aggregation.total_gpu_count, 0)
        self.assertIn("sh02-01n61", aggregation.matched_node_names)
        self.assertEqual(aggregation.missing_node_names, ["missing-node"])


if __name__ == "__main__":
    unittest.main()
