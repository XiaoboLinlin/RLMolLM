import random

# Dataset statistics - BERT TOKENIZER TOKEN COUNTS
# Uses actual project tokenizer setup with tokenizer_type='bert'
DATASET_STATS = {
    "moses": {
        2: 0.000330,
        3: 0.000330,
        4: 0.000660,
        5: 0.003450,
        6: 0.008380,
        7: 0.013010,
        8: 0.011380,
        9: 0.016090,
        10: 0.055360,
        11: 0.056720,
        12: 0.051400,
        13: 0.028180,
        14: 0.092100,
        15: 0.104800,
        16: 0.067790,
        17: 0.034120,
        18: 0.071980,
        19: 0.105320,
        20: 0.046640,
        21: 0.029230,
        22: 0.034990,
        23: 0.065380,
        24: 0.024010,
        25: 0.014420,
        26: 0.013880,
        27: 0.021270,
        28: 0.008360,
        29: 0.005120,
        30: 0.004350,
        31: 0.004820,
        32: 0.002090,
        33: 0.001360,
        34: 0.000980,
        35: 0.000810,
        36: 0.000330,
        37: 0.000200,
        38: 0.000120,
        39: 0.000150,
        40: 0.000020,
        41: 0.000030,
        42: 0.000010,
        43: 0.000030,
    },
    "guacamol": {
        1: 0.000270,
        2: 0.000650,
        3: 0.001000,
        4: 0.000910,
        5: 0.004190,
        6: 0.005890,
        7: 0.007800,
        8: 0.006600,
        9: 0.013930,
        10: 0.021210,
        11: 0.025000,
        12: 0.021950,
        13: 0.024630,
        14: 0.037490,
        15: 0.041500,
        16: 0.036610,
        17: 0.031970,
        18: 0.042680,
        19: 0.049970,
        20: 0.043400,
        21: 0.035320,
        22: 0.039260,
        23: 0.047330,
        24: 0.039290,
        25: 0.032900,
        26: 0.031730,
        27: 0.036520,
        28: 0.032510,
        29: 0.026730,
        30: 0.024070,
        31: 0.024960,
        32: 0.022950,
        33: 0.019820,
        34: 0.016930,
        35: 0.016070,
        36: 0.015520,
        37: 0.013310,
        38: 0.011660,
        39: 0.010390,
        40: 0.008890,
        41: 0.008240,
        42: 0.006680,
        43: 0.006090,
        44: 0.005850,
        45: 0.005500,
        46: 0.004720,
        47: 0.004530,
        48: 0.003740,
        49: 0.002980,
        50: 0.002910,
        51: 0.002880,
        52: 0.002440,
        53: 0.002140,
        54: 0.001980,
        55: 0.001420,
        56: 0.002070,
        57: 0.001500,
        58: 0.001220,
        59: 0.001120,
        60: 0.001120,
        61: 0.000780,
        62: 0.000730,
        63: 0.000540,
        64: 0.000560,
        65: 0.000770,
        66: 0.000530,
        67: 0.000540,
        68: 0.000400,
        69: 0.000480,
        70: 0.000300,
        71: 0.000290,
        72: 0.000230,
        73: 0.000170,
        74: 0.000170,
        75: 0.000150,
        76: 0.000100,
        77: 0.000090,
        78: 0.000060,
        79: 0.000060,
        80: 0.000020,
        81: 0.000010,
        82: 0.000010,
        83: 0.000040,
        85: 0.000020,
        91: 0.000010,
    },
}

def sample_token_count(dataset_name: str) -> int:
    """
    Samples a token count based on the empirical distribution of the specified dataset.
    Uses BERT tokenizer token counts (actual project setup).
    
    Args:
        dataset_name: Name of the dataset ('moses' or 'guacamol')
        
    Returns:
        int: Sampled token count
    """
    if dataset_name not in DATASET_STATS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_STATS.keys())}")
    
    distribution = DATASET_STATS[dataset_name]
    token_counts = list(distribution.keys())
    probabilities = list(distribution.values())
    
    return random.choices(token_counts, weights=probabilities, k=1)[0]

if __name__ == "__main__":
    # Example usage and testing
    print("Testing BERT token count sampling:")
    moses_samples = [sample_token_count('moses') for _ in range(5)]
    guacamol_samples = [sample_token_count('guacamol') for _ in range(5)]
    print(f"Moses: {moses_samples}")
    print(f"Guacamol: {guacamol_samples}")
