"""Regression tests for SBATCH walltime parsing."""

from __future__ import annotations

import unittest

from sbatch_parser import parse_memory_to_gigabytes, parse_sbatch_text, parse_walltime_hours


class TestParseWalltimeHours(unittest.TestCase):
    def test_minutes_seconds_format_is_sub_hour(self) -> None:
        parsed_hours = parse_walltime_hours("10:00")
        self.assertIsNotNone(parsed_hours)
        self.assertAlmostEqual(parsed_hours, 10.0 / 60.0, places=8)

    def test_single_segment_is_minutes(self) -> None:
        parsed_hours = parse_walltime_hours("59")
        self.assertIsNotNone(parsed_hours)
        self.assertAlmostEqual(parsed_hours, 59.0 / 60.0, places=8)

    def test_three_segments_are_hours_minutes_seconds(self) -> None:
        parsed_hours = parse_walltime_hours("01:00:00")
        self.assertIsNotNone(parsed_hours)
        self.assertAlmostEqual(parsed_hours, 1.0, places=8)

    def test_day_hour_minute_format(self) -> None:
        parsed_hours = parse_walltime_hours("1-00:00")
        self.assertIsNotNone(parsed_hours)
        self.assertAlmostEqual(parsed_hours, 24.0, places=8)

    def test_day_hour_format_without_minutes(self) -> None:
        parsed_hours = parse_walltime_hours("0-00")
        self.assertIsNotNone(parsed_hours)
        self.assertAlmostEqual(parsed_hours, 0.0, places=8)

    def test_minutes_seconds_with_leading_zeros(self) -> None:
        parsed_hours = parse_walltime_hours("00:30")
        self.assertIsNotNone(parsed_hours)
        self.assertAlmostEqual(parsed_hours, 30.0 / 3600.0, places=8)

    def test_day_hour_format_multiple_days(self) -> None:
        parsed_hours = parse_walltime_hours("2-12")
        self.assertIsNotNone(parsed_hours)
        self.assertAlmostEqual(parsed_hours, 60.0, places=8)

    def test_invalid_format_returns_none(self) -> None:
        self.assertIsNone(parse_walltime_hours("1::00"))

    def test_zero_day_prefix_is_hours_not_minutes(self) -> None:
        parsed_hours = parse_walltime_hours("0-12")
        self.assertIsNotNone(parsed_hours)
        self.assertAlmostEqual(parsed_hours, 12.0, places=8)

    def test_one_day_plus_hours(self) -> None:
        parsed_hours = parse_walltime_hours("1-12")
        self.assertIsNotNone(parsed_hours)
        self.assertAlmostEqual(parsed_hours, 36.0, places=8)

    def test_bare_number_is_minutes(self) -> None:
        parsed_hours = parse_walltime_hours("30")
        self.assertIsNotNone(parsed_hours)
        self.assertAlmostEqual(parsed_hours, 0.5, places=8)

    def test_two_segments_without_day_prefix_are_minutes_seconds(self) -> None:
        parsed_hours = parse_walltime_hours("30:00")
        self.assertIsNotNone(parsed_hours)
        self.assertAlmostEqual(parsed_hours, 0.5, places=8)

    def test_hours_minutes_seconds(self) -> None:
        parsed_hours = parse_walltime_hours("1:30:00")
        self.assertIsNotNone(parsed_hours)
        self.assertAlmostEqual(parsed_hours, 1.5, places=8)

    def test_day_hour_minute_second_format(self) -> None:
        parsed_hours = parse_walltime_hours("1-02:30:00")
        self.assertIsNotNone(parsed_hours)
        self.assertAlmostEqual(parsed_hours, 26.5, places=8)


class TestParseMemoryToGigabytes(unittest.TestCase):
    def test_unitless_memory_defaults_to_megabytes(self) -> None:
        parsed_gigabytes = parse_memory_to_gigabytes("4000")
        self.assertIsNotNone(parsed_gigabytes)
        self.assertAlmostEqual(parsed_gigabytes, 4000.0 / 1024.0, places=8)

    def test_explicit_gigabyte_unit_is_unchanged(self) -> None:
        parsed_gigabytes = parse_memory_to_gigabytes("4G")
        self.assertIsNotNone(parsed_gigabytes)
        self.assertAlmostEqual(parsed_gigabytes, 4.0, places=8)


class TestParseSbatchTextWalltime(unittest.TestCase):
    def test_sbatch_time_directive_parses_as_minutes_seconds(self) -> None:
        sbatch_text = "\n".join(
            [
                "#!/bin/bash",
                "#SBATCH --job-name=collect_stats",
                "#SBATCH --time=10:00",
            ]
        )
        parameters = parse_sbatch_text(sbatch_text)
        self.assertIsNotNone(parameters.walltime_hours)
        self.assertAlmostEqual(parameters.walltime_hours, 10.0 / 60.0, places=8)


class TestParseSbatchExactFlagMatching(unittest.TestCase):
    def test_mem_per_cpu_does_not_set_total_memory(self) -> None:
        parameters = parse_sbatch_text("#SBATCH --mem-per-cpu=4G")
        self.assertAlmostEqual(parameters.memory_gigabytes_per_cpu, 4.0, places=8)
        self.assertIsNone(parameters.memory_gigabytes_total)

    def test_ntasks_per_node_does_not_set_ntasks(self) -> None:
        parameters = parse_sbatch_text("#SBATCH --ntasks-per-node=8")
        self.assertEqual(parameters.ntasks_per_node, 8)
        self.assertIsNone(parameters.task_count)

    def test_unitless_mem_directive_is_megabytes(self) -> None:
        parameters = parse_sbatch_text("#SBATCH --mem=4000")
        self.assertIsNotNone(parameters.memory_gigabytes_total)
        self.assertAlmostEqual(parameters.memory_gigabytes_total, 4000.0 / 1024.0, places=8)


class TestParseSbatchShortFlags(unittest.TestCase):
    def test_spaced_short_flags(self) -> None:
        sbatch_text = "\n".join(
            [
                "#!/bin/bash",
                "#SBATCH -p gpu",
                "#SBATCH -N 2",
                "#SBATCH -n 4",
                "#SBATCH -c 8",
                "#SBATCH -t 2:00:00",
                "#SBATCH -w sh02-01n61",
                "#SBATCH -G 2",
            ]
        )
        parameters = parse_sbatch_text(sbatch_text)
        self.assertEqual(parameters.partition_name, "gpu")
        self.assertEqual(parameters.node_count, 2)
        self.assertEqual(parameters.task_count, 4)
        self.assertEqual(parameters.cpu_cores_per_task, 8)
        self.assertAlmostEqual(parameters.walltime_hours, 2.0, places=8)
        self.assertEqual(parameters.nodelist, "sh02-01n61")
        self.assertEqual(parameters.gpu_count, 2)

    def test_attached_short_time_and_gpu_flags(self) -> None:
        sbatch_text = "\n".join(
            [
                "#SBATCH -t2:00:00",
                "#SBATCH -G2",
                "#SBATCH --gpus=3",
            ]
        )
        parameters = parse_sbatch_text(sbatch_text)
        self.assertAlmostEqual(parameters.walltime_hours, 2.0, places=8)
        self.assertEqual(parameters.gpu_count, 3)


if __name__ == "__main__":
    unittest.main()
