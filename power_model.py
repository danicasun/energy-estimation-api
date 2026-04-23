"""Power consumption and emissions calculation model."""

from typing import Dict, List, Optional

from energy_constants import POWER_USAGE_EFFECTIVENESS


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
        Facility energy consumption in kWh (IT energy multiplied by PUE)
    """
    it_energy_kwh = (power_watts * hours) / 1000.0
    return it_energy_kwh * POWER_USAGE_EFFECTIVENESS


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


def calculate_aggregate_node_power(
    allocated_node_cpu_cores: List[int],
    total_memory_gigabytes: float,
) -> float:
    """
    Calculate aggregate power across allocated nodes in Watts.

    Memory in GB is distributed proportional to node CPU core counts.
    """
    if not allocated_node_cpu_cores:
        return calculate_power(0, total_memory_gigabytes)

    sanitized_cpu_core_counts = [max(int(cpu_cores), 0) for cpu_cores in allocated_node_cpu_cores]
    total_allocated_cpu_cores = sum(sanitized_cpu_core_counts)
    allocated_node_count = len(sanitized_cpu_core_counts)

    if total_allocated_cpu_cores <= 0:
        memory_per_node_gigabytes = total_memory_gigabytes / max(allocated_node_count, 1)
        return sum(calculate_power(0, memory_per_node_gigabytes) for _ in sanitized_cpu_core_counts)

    aggregate_power_watts = 0.0
    for node_cpu_core_count in sanitized_cpu_core_counts:
        memory_fraction = node_cpu_core_count / total_allocated_cpu_cores
        node_memory_gigabytes = total_memory_gigabytes * memory_fraction
        aggregate_power_watts += calculate_power(node_cpu_core_count, node_memory_gigabytes)
    return aggregate_power_watts


def estimate_emissions(
    cpus: int,
    mem_gb: float,
    hours: float,
    carbon_intensity: float,
    allocated_node_cpu_cores: Optional[List[int]] = None,
) -> Dict[str, float]:
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
    power = (
        calculate_aggregate_node_power(allocated_node_cpu_cores, mem_gb)
        if allocated_node_cpu_cores
        else calculate_power(cpus, mem_gb)
    )
    energy = calculate_energy(power, hours)
    emissions_g = calculate_emissions(energy, carbon_intensity)
    
    return {
        "power_watts": power,
        "energy_kwh": energy,
        "emissions_gco2e": emissions_g,
        "emissions_kgco2e": emissions_g / 1000.0,
    }

