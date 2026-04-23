"""Integration-oriented tests for API parsing and node lookup behavior."""

from __future__ import annotations

import unittest
from unittest.mock import patch

try:
    from job_prediction_api import _build_prediction_response, _parse_request_payload
except ModuleNotFoundError:  # pragma: no cover - dependency optional in local test env
    _build_prediction_response = None
    _parse_request_payload = None


@unittest.skipIf(_build_prediction_response is None or _parse_request_payload is None, "fastapi not installed")
class TestJobPredictionApiNodeLookup(unittest.TestCase):
    @patch("slurm_runtime._expand_with_scontrol", return_value=[])
    @patch("job_prediction_api.get_slurm_env_vars", return_value={"nodelist": None})
    def test_parse_payload_infers_cpu_cores_from_node_inventory(
        self,
        _: object,
        __: object,
    ) -> None:
        payload = {
            "sbatchText": "\n".join(
                [
                    "#!/bin/bash",
                    "#SBATCH --nodelist=sh02-01n[61-62]",
                    "#SBATCH --time=10:00",
                ]
            ),
            "parameters": {},
        }

        parsed_inputs = _parse_request_payload(payload)

        self.assertEqual(parsed_inputs["allocated_node_names"], ["sh02-01n61", "sh02-01n62"])
        self.assertEqual(parsed_inputs["allocated_node_cpu_cores"], [20, 20])
        self.assertEqual(parsed_inputs["cpu_cores"], 40)
        self.assertEqual(parsed_inputs["node_count"], 2)

    @patch("job_prediction_api.get_carbon_intensity", return_value=400.0)
    def test_prediction_response_includes_missing_inventory_note(self, _: object) -> None:
        inputs = {
            "cpu_cores": 1,
            "gpu_count": 0,
            "node_count": 1,
            "partition_name": "normal",
            "memory_gigabytes": 1.0,
            "walltime_hours": 1.0,
            "zone": "US-CA",
            "resolved_nodelist": "unknown-node",
            "allocated_node_names": ["unknown-node"],
            "allocated_node_cpu_cores": None,
            "missing_inventory_node_names": ["unknown-node"],
        }

        response = _build_prediction_response(inputs)
        notes_text = " ".join(response["notes"])

        self.assertIn("missing for: unknown-node", notes_text)


if __name__ == "__main__":
    unittest.main()
