"""
All available ADMET property configurations.
Extracted from admet-ai package and literature.

Each property has:
- range: [min, max] for normalization
- preferred_value: target value for optimization (closer is better)
  - If higher is better: use max value
  - If lower is better: use min value  
  - If specific target: use that value
"""

# All available ADMET and molecular properties with their configurations
PROPERTY_CONFIGS = {
    # Absorption & Permeability (higher is better)
    "HIA_Hou": {
        "range": [0, 1],
        "preferred_value": 1.0
    },
    "Caco2_Wang": {
        "range": [-8, 2],
        "preferred_value": 2.0
    },
    "PAMPA_NCATS": {
        "range": [0, 1],
        "preferred_value": 1.0
    },
    "Bioavailability_Ma": {
        "range": [0, 1],
        "preferred_value": 1.0
    },
    
    # Metabolism (CYP Enzyme Activity) - lower is better (minimize substrate activity)
    "CYP3A4_Substrate_CarbonMangels": {
        "range": [0, 1],
        "preferred_value": 0.0
    },
    "CYP2D6_Substrate_CarbonMangels": {
        "range": [0, 1],
        "preferred_value": 0.0
    },
    "CYP2C9_Substrate_CarbonMangels": {
        "range": [0, 1],
        "preferred_value": 0.0
    },
    "CYP1A2_Veith": {
        "range": [0, 1],
        "preferred_value": 0.0
    },
    "CYP2C19_Veith": {
        "range": [0, 1],
        "preferred_value": 0.0
    },
    "CYP2C9_Veith": {
        "range": [0, 1],
        "preferred_value": 0.0
    },
    "CYP2D6_Veith": {
        "range": [0, 1],
        "preferred_value": 0.0
    },
    "CYP3A4_Veith": {
        "range": [0, 1],
        "preferred_value": 0.0
    },
    "Clearance_Hepatocyte_AZ": {
        "range": [-10, 10],
        "preferred_value": 0.0
    },
    "Clearance_Microsome_AZ": {
        "range": [-10, 10],
        "preferred_value": 0.0
    },
    
    # Distribution (Blood-Brain Barrier & Plasma Protein Binding)
    "BBB_Martins": {
        "range": [0, 1],
        "preferred_value": 0.0  # Lower is better (unless targeting CNS)
    },
    "PPBR_AZ": {
        "range": [0, 100],
        "preferred_value": 50.0  # Moderate binding
    },
    "VDss_Lombardo": {
        "range": [-5, 5],
        "preferred_value": 0.0  # Moderate distribution
    },
    
    # Excretion & Half-life
    "Half_Life_Obach": {
        "range": [-30, 30],
        "preferred_value": 0.0  # Moderate half-life
    },
    
    # Toxicity & Safety (lower is better)
    "hERG": {
        "range": [0, 1],
        "preferred_value": 0.0
    },
    "AMES": {
        "range": [0, 1],
        "preferred_value": 0.0
    },
    "DILI": {
        "range": [0, 1],
        "preferred_value": 0.0
    },
    "ClinTox": {
        "range": [0, 1],
        "preferred_value": 0.0
    },
    "Carcinogens_Lagunin": {
        "range": [0, 1],
        "preferred_value": 0.0
    },
    "Skin_Reaction": {
        "range": [0, 1],
        "preferred_value": 0.0
    },
    "LD50_Zhu": {
        "range": [0, 5],
        "preferred_value": 5.0  # Higher is better (less toxic)
    },
    
    # Physicochemical Properties
    "logP": {
        "range": [-3, 7],
        "preferred_value": 2.0  # Target moderate lipophilicity
    },
    "tpsa": {
        "range": [0, 200],
        "preferred_value": 80.0  # Target moderate TPSA
    },
    "Solubility_AqSolDB": {
        "range": [-10, 2],
        "preferred_value": -1.0  # Target good solubility
    },
    "Lipinski": {
        "range": [0, 4],
        "preferred_value": 0.0  # Minimize violations
    },
    "QED": {
        "range": [0, 1],
        "preferred_value": 1.0  # Maximize drug-likeness
    },
    "HydrationFreeEnergy_FreeSolv": {
        "range": [-30, 10],
        "preferred_value": -7.5  # Target moderate hydration
    },
    "Lipophilicity_AstraZeneca": {
        "range": [-3, 3],
        "preferred_value": 0.5  # Target moderate lipophilicity
    },
    
    # Other Properties
    "molecular_weight": {
        "range": [0, 900],
        "preferred_value": 300.0  # Target typical drug MW
    },
    "hydrogen_bond_acceptors": {
        "range": [0, 15],
        "preferred_value": 5.0  # Moderate H-bond acceptors
    },
    "hydrogen_bond_donors": {
        "range": [0, 8],
        "preferred_value": 2.5  # Moderate H-bond donors
    },
    "stereo_centers": {
        "range": [0, 10],
        "preferred_value": 1.5  # Few stereocenters
    },
    "Pgp_Broccatelli": {
        "range": [0, 1],
        "preferred_value": 0.0  # Minimize P-gp substrate activity
    },
    "NR-AR-LBD": {
        "range": [0, 1],
        "preferred_value": 0.0
    },
    "NR-AR": {
        "range": [0, 1],
        "preferred_value": 0.0
    },
    "NR-AhR": {
        "range": [0, 1],
        "preferred_value": 0.0
    },
    "NR-Aromatase": {
        "range": [0, 1],
        "preferred_value": 0.0
    },
    "NR-ER-LBD": {
        "range": [0, 1],
        "preferred_value": 0.0
    },
    "NR-ER": {
        "range": [0, 1],
        "preferred_value": 0.0
    },
    "NR-PPAR-gamma": {
        "range": [0, 1],
        "preferred_value": 0.0
    },
    "SR-ARE": {
        "range": [0, 1],
        "preferred_value": 0.0
    },
    "SR-ATAD5": {
        "range": [0, 1],
        "preferred_value": 0.0
    },
    "SR-HSE": {
        "range": [0, 1],
        "preferred_value": 0.0
    },
    "SR-MMP": {
        "range": [0, 1],
        "preferred_value": 0.0
    },
    "SR-p53": {
        "range": [0, 1],
        "preferred_value": 0.0
    }
}


def get_property_config(property_name):
    """
    Get configuration for a specific ADMET/molecular property.
    
    Args:
        property_name: Name of the property (e.g., 'PAMPA_NCATS', 'hERG')
        
    Returns:
        Dict with 'range' and 'preferred_value' keys, or None if not found
    """
    return PROPERTY_CONFIGS.get(property_name)


def merge_property_configs(user_config, additional_properties):
    """
    Merge user's property_config with configurations from PROPERTY_CONFIGS
    for additional properties.
    
    Args:
        user_config: User's existing property_config dict
        additional_properties: List of property names to add from PROPERTY_CONFIGS
        
    Returns:
        Merged property_config dict
    """
    merged_config = user_config.copy()
    
    for prop_name in additional_properties:
        if prop_name not in merged_config and prop_name in PROPERTY_CONFIGS:
            merged_config[prop_name] = PROPERTY_CONFIGS[prop_name].copy()
    
    return merged_config


def get_all_properties():
    """Get list of all available property names."""
    return list(PROPERTY_CONFIGS.keys())


if __name__ == "__main__":
    # Quick test
    print(f"Total properties available: {len(PROPERTY_CONFIGS)}")
    print(f"\nSample - PAMPA_NCATS: {get_property_config('PAMPA_NCATS')}")
    print(f"Sample - hERG: {get_property_config('hERG')}")
    print(f"Sample - logP: {get_property_config('logP')}")
    
    # Test merge
    user_config = {
        'synth': {'range': [0, 8], 'higher_is_better': False},
        'drug': {'range': [0, 1], 'higher_is_better': True}
    }
    merged = merge_property_configs(user_config, ['PAMPA_NCATS', 'AMES'])
    print(f"\nMerged config has {len(merged)} properties")
    print(f"Added: {[k for k in merged.keys() if k not in user_config]}")
