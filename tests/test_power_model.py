"""Regression tests for energy and emissions estimation."""

from __future__ import annotations

import unittest

from power_model import (
    MEM_WATTS_PER_GB,
    CPU_WATTS_PER_CORE,
    calculate_emissions,
    calculate_energy,
    calculate_power,
    estimate_emissions,
)


class TestPowerModelUnits(unittest.TestCase):
    def test_calculate_power_watts(self) -> None:
        cpu_cores = 8  # CPU cores
        memory_gigabytes = 32.0  # GB

        expected_power_watts = (cpu_cores * CPU_WATTS_PER_CORE) + (
            memory_gigabytes * MEM_WATTS_PER_GB
        )
        self.assertAlmostEqual(calculate_power(cpu_cores, memory_gigabytes), expected_power_watts, places=8)

    def test_calculate_energy_kwh(self) -> None:
        power_watts = 100.0  # W
        walltime_hours = 2.5  # h

        expected_energy_kwh = 0.25  # kWh
        self.assertAlmostEqual(calculate_energy(power_watts, walltime_hours), expected_energy_kwh, places=8)

    def test_calculate_emissions_grams(self) -> None:
        energy_kwh = 0.25  # kWh
        carbon_intensity_gco2e_per_kwh = 400.0  # gCO2e/kWh

        expected_emissions_grams = 100.0  # gCO2e
        self.assertAlmostEqual(
            calculate_emissions(energy_kwh, carbon_intensity_gco2e_per_kwh),
            expected_emissions_grams,
            places=8,
        )


class TestEstimateEmissionsPipeline(unittest.TestCase):
    def test_estimate_emissions_pipeline_outputs_consistent_units(self) -> None:
        cpu_cores = 4  # CPU cores
        memory_gigabytes = 16.0  # GB
        walltime_hours = 1.5  # h
        carbon_intensity_gco2e_per_kwh = 300.0  # gCO2e/kWh

        result = estimate_emissions(
            cpus=cpu_cores,
            mem_gb=memory_gigabytes,
            hours=walltime_hours,
            carbon_intensity=carbon_intensity_gco2e_per_kwh,
        )

        expected_power_watts = (cpu_cores * CPU_WATTS_PER_CORE) + (
            memory_gigabytes * MEM_WATTS_PER_GB
        )
        expected_energy_kwh = (expected_power_watts * walltime_hours) / 1000.0
        expected_emissions_gco2e = expected_energy_kwh * carbon_intensity_gco2e_per_kwh
        expected_emissions_kgco2e = expected_emissions_gco2e / 1000.0

        self.assertAlmostEqual(result["power_watts"], expected_power_watts, places=8)
        self.assertAlmostEqual(result["energy_kwh"], expected_energy_kwh, places=8)
        self.assertAlmostEqual(result["emissions_gco2e"], expected_emissions_gco2e, places=8)
        self.assertAlmostEqual(result["emissions_kgco2e"], expected_emissions_kgco2e, places=8)


if __name__ == "__main__":
    unittest.main()
