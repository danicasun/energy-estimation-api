"""Regression tests for SBATCH walltime parsing."""

from __future__ import annotations

import unittest

from sbatch_parser import parse_sbatch_text, parse_walltime_hours


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


if __name__ == "__main__":
    unittest.main()
