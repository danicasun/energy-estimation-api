"""Query ElectricityMaps API for carbon intensity data."""

import os
import requests
from typing import Optional
from datetime import datetime


API_TOKEN = "xiXORcxkXP3ggwy4cc2v"


def get_carbon_intensity(zone: str, api_key: Optional[str] = None, timestamp: Optional[datetime] = None) -> Optional[float]:
    """
    Get current or historical carbon intensity for a zone.
    
    Args:
        zone: ElectricityMaps zone code (e.g., "US-CAL-CISO")
        api_key: ElectricityMaps API key (defaults to ELECTRICITYMAPS_API_KEY env var, then DEFAULT_API_TOKEN)
        timestamp: Optional datetime for historical data (defaults to current time)
        
    Returns:
        Carbon intensity in gCO₂e/kWh, or None if unavailable
    """
    if not api_key:
        api_key = os.environ.get("ELECTRICITYMAPS_API_KEY")
    
    if not api_key:
        api_key = API_TOKEN
    
    if not api_key:
        print("Warning: No API key available, using fallback value")
        return get_fallback_carbon_intensity(zone)
    
    try:
        if timestamp:
            # Historical data endpoint
            url = "https://api.electricitymaps.com/v3/carbon-intensity/history"
            params = {
                "zone": zone,
                "datetime": timestamp.isoformat()
            }
        else:
            # Current/latest data endpoint
            url = f"https://api.electricitymaps.com/v3/carbon-intensity/latest"
            params = {"zone": zone}
        
        headers = {"auth-token": api_key}
        
        response = requests.get(url, headers=headers, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            # Extract carbonIntensity from response
            if "carbonIntensity" in data:
                return float(data["carbonIntensity"])
            elif "data" in data and len(data["data"]) > 0:
                # Historical data format
                return float(data["data"][-1].get("carbonIntensity", 0))
        else:
            print(f"Warning: ElectricityMaps API error ({response.status_code}), using fallback")
            return get_fallback_carbon_intensity(zone)
    
    except (requests.RequestException, KeyError, ValueError) as e:
        print(f"Warning: Failed to fetch carbon intensity ({e}), using fallback")
        return get_fallback_carbon_intensity(zone)


def get_fallback_carbon_intensity(zone: str) -> float:
    """
    Get fallback carbon intensity value when API is unavailable.
    
    Uses average values for common zones.
    
    Args:
        zone: ElectricityMaps zone code
        
    Returns:
        Fallback carbon intensity in gCO₂e/kWh
    """
    # Average carbon intensity values (gCO₂e/kWh) for common zones
    fallback_values = {
        "US-CAL-CISO": 250.0,  # California average
        "US-NY-ISONE": 200.0,  # New York average
        "US-TX-ERCOT": 400.0,  # Texas average
        "US-PJM": 350.0,  # PJM average
    }
    
    # Default fallback
    default = 300.0
    
    return fallback_values.get(zone, default)

