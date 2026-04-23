"""Tests for Slurm nodelist expansion utilities."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from slurm_runtime import expand_slurm_nodelist


class TestExpandSlurmNodelist(unittest.TestCase):
    @patch("slurm_runtime._expand_with_scontrol", return_value=[])
    def test_expand_single_node_without_brackets(self, _: object) -> None:
        self.assertEqual(expand_slurm_nodelist("sh02-01n61"), ["sh02-01n61"])

    @patch("slurm_runtime._expand_with_scontrol", return_value=[])
    def test_expand_bracket_range(self, _: object) -> None:
        expanded_nodes = expand_slurm_nodelist("sh02-01n[61-63]")
        self.assertEqual(expanded_nodes, ["sh02-01n61", "sh02-01n62", "sh02-01n63"])

    @patch("slurm_runtime._expand_with_scontrol", return_value=[])
    def test_expand_multiple_tokens_with_range_and_single(self, _: object) -> None:
        expanded_nodes = expand_slurm_nodelist("sh02-01n[61-62],sh02-02n01")
        self.assertEqual(expanded_nodes, ["sh02-01n61", "sh02-01n62", "sh02-02n01"])


if __name__ == "__main__":
    unittest.main()
