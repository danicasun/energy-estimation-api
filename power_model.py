"""Power consumption and emissions calculation model."""

from typing import Optional


# Power model constants (Watts)
CPU_WATTS_PER_CORE = 10.0  # 10 W per CPU core
MEM_WATTS_PER_GB = 0.372   # 0.372 W per GB of memory


def calculate_power(cpus: int, mem_gb: float) -> float:
    """
    Calculate total power consumption in Watts.
    
    Uses simple linear model:
    - CPU: 10 W per core
    - Memory: 0.372 W per GB
    
    Args:
        cpus: Number of CPU cores
        mem_gb: Memory in GB
        
    Returns:
        Total power consumption in Watts
    """
    cpu_power = cpus * CPU_WATTS_PER_CORE
    mem_power = mem_gb * MEM_WATTS_PER_GB
    return cpu_power + mem_power


def calculate_energy(power_watts: float, hours: float) -> float:
    """
    Calculate energy consumption in kWh.
    
    Args:
        power_watts: Power consumption in Watts
        hours: Duration in hours
        
    Returns:
        Energy consumption in kWh
    """
    return (power_watts * hours) / 1000.0


def calculate_emissions(energy_kwh: float, carbon_intensity: float) -> float:
    """
    Calculate CO₂e emissions in grams.
    
    Args:
        energy_kwh: Energy consumption in kWh
        carbon_intensity: Carbon intensity in gCO₂e/kWh
        
    Returns:
        Emissions in grams CO₂e
    """
    return energy_kwh * carbon_intensity


def estimate_emissions(
    cpus: int,
    mem_gb: float,
    hours: float,
    carbon_intensity: float
) -> dict:
    """
    Complete emissions estimation pipeline.
    
    Args:
        cpus: Number of CPU cores
        mem_gb: Memory in GB
        hours: Duration in hours
        carbon_intensity: Carbon intensity in gCO₂e/kWh
        
    Returns:
        Dictionary with:
        - power_watts: Power consumption (W)
        - energy_kwh: Energy consumption (kWh)
        - emissions_gco2e: Emissions (g CO₂e)
        - emissions_kgco2e: Emissions (kg CO₂e)
    """
    power = calculate_power(cpus, mem_gb)
    energy = calculate_energy(power, hours)
    emissions_g = calculate_emissions(energy, carbon_intensity)
    
    return {
        "power_watts": power,
        "energy_kwh": energy,
        "emissions_gco2e": emissions_g,
        "emissions_kgco2e": emissions_g / 1000.0,
    }

