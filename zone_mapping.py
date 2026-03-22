"""Map compute node prefixes to ElectricityMaps zones."""

from typing import Dict, Optional


# Default mapping: node prefix -> ElectricityMaps zone
DEFAULT_ZONE_MAPPING: Dict[str, str] = {
    "sh": "US-CAL-CISO",  # Sherlock cluster
    "node": "US-CAL-CISO",  # Generic node prefix
    "compute": "US-CAL-CISO",  # Generic compute prefix
}


def get_zone_for_node_prefix(prefix: Optional[str], custom_mapping: Optional[Dict[str, str]] = None) -> str:
    """
    Get ElectricityMaps zone for a node prefix.
    
    Args:
        prefix: Node prefix (e.g., "sh", "node")
        custom_mapping: Optional custom mapping dictionary
        
    Returns:
        ElectricityMaps zone code (defaults to US-CAL-CISO)
    """
    if not prefix:
        return "US-CAL-CISO"  # Default zone
    
    # Use custom mapping if provided, otherwise use default
    mapping = custom_mapping if custom_mapping else DEFAULT_ZONE_MAPPING
    
    # Try exact match first
    if prefix in mapping:
        return mapping[prefix]
    
    # Try case-insensitive match
    prefix_lower = prefix.lower()
    for key, zone in mapping.items():
        if key.lower() == prefix_lower:
            return zone
    
    # Default fallback
    return "US-CAL-CISO"


def set_custom_zone_mapping(mapping: Dict[str, str]) -> None:
    """
    Set a custom zone mapping.
    
    Args:
        mapping: Dictionary mapping node prefixes to zones
    """
    global DEFAULT_ZONE_MAPPING
    DEFAULT_ZONE_MAPPING.update(mapping)
