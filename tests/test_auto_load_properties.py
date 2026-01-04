"""
Test that ADMET properties from options_range.json can be used without manual config.
"""
import os
import pytest
from rlmollm import RLMolLMGenerator


def test_auto_load_admet_property(generator):
    """Test that we can use PAMPA_NCATS without adding it to config manually."""
    output_dir = "tests/test_output/auto_load_admet"
    os.makedirs(output_dir, exist_ok=True)
    
    # Use PAMPA_NCATS which is in options_range.json but NOT in no_scaffold_2_moses_admet.json
    target_properties = {
        'qed': 1.0,
        'sa_score': 1.0,
        'PAMPA_NCATS': 1.0,  # This should auto-load from options_range.json!
    }
    
    print("\n" + "="*70)
    print("Testing auto-load of PAMPA_NCATS property...")
    print("="*70)
    
    molecules_df = generator.optimize(
        target_properties=target_properties,
        initial_population_file="training_output_moses/no_scaffold_2_moses/2000_initial/initial_population.csv",
        population_size=50,
        generations=2,
        output_dir=output_dir,
        return_dataframe=True
    )
    
    # Verify PAMPA_NCATS appears in output
    assert 'PAMPA_NCATS' in molecules_df.columns, "PAMPA_NCATS should be in output columns"
    assert 'PAMPA_NCATS_raw' in molecules_df.columns, "PAMPA_NCATS_raw should be in output columns"
    
    # Check that values are within expected range [0, 1]
    assert molecules_df['PAMPA_NCATS_raw'].min() >= 0, "PAMPA_NCATS_raw should be >= 0"
    assert molecules_df['PAMPA_NCATS_raw'].max() <= 1, "PAMPA_NCATS_raw should be <= 1"
    
    # Check that normalized values are in [0, 1]
    assert molecules_df['PAMPA_NCATS'].min() >= 0, "PAMPA_NCATS normalized should be >= 0"
    assert molecules_df['PAMPA_NCATS'].max() <= 1, "PAMPA_NCATS normalized should be <= 1"
    
    print(f"\n✓ PAMPA_NCATS successfully auto-loaded and calculated!")
    print(f"   Raw range: [{molecules_df['PAMPA_NCATS_raw'].min():.3f}, {molecules_df['PAMPA_NCATS_raw'].max():.3f}]")
    print(f"   Normalized range: [{molecules_df['PAMPA_NCATS'].min():.3f}, {molecules_df['PAMPA_NCATS'].max():.3f}]")
    
    # Read final CSV to verify it was saved correctly
    import pandas as pd
    csv_df = pd.read_csv(f"{output_dir}/final_population.csv")
    assert 'PAMPA_NCATS' in csv_df.columns, "PAMPA_NCATS should be in CSV"
    assert 'PAMPA_NCATS_raw' in csv_df.columns, "PAMPA_NCATS_raw should be in CSV"
    
    print(f"✓ PAMPA_NCATS correctly saved to CSV")


def test_multiple_auto_load_properties(generator):
    """Test that we can use multiple auto-loaded properties at once."""
    output_dir = "tests/test_output/multi_auto_load"
    os.makedirs(output_dir, exist_ok=True)
    
    # Use multiple properties from options_range.json
    target_properties = {
        'qed': 1.0,
        'PAMPA_NCATS': 1.0,      # Not in config
        'AMES': 0.0,              # Not in config (toxicity - lower is better)
        'Solubility_AqSolDB': -2.0, # Not in config
    }
    
    print("\n" + "="*70)
    print("Testing multiple auto-loaded ADMET properties...")
    print(f"Properties: {list(target_properties.keys())}")
    print("="*70)
    
    molecules_df = generator.optimize(
        target_properties=target_properties,
        initial_population_file="training_output_moses/no_scaffold_2_moses/2000_initial/initial_population.csv",
        population_size=50,
        generations=2,
        output_dir=output_dir,
        return_dataframe=True
    )
    
    # Verify all properties appear in output
    for prop in ['PAMPA_NCATS', 'AMES', 'Solubility_AqSolDB']:
        assert prop in molecules_df.columns, f"{prop} should be in output columns"
        assert f"{prop}_raw" in molecules_df.columns, f"{prop}_raw should be in output columns"
        print(f"✓ {prop} successfully auto-loaded")
        print(f"   Raw range: [{molecules_df[f'{prop}_raw'].min():.3f}, {molecules_df[f'{prop}_raw'].max():.3f}]")


def test_mixed_config_and_auto_load(generator):
    """Test mixing properties in config with auto-loaded ones."""
    output_dir = "tests/test_output/mixed_properties"
    os.makedirs(output_dir, exist_ok=True)
    
    # Mix properties that ARE in config (hERG, Caco2_Wang) with ones that are NOT (PAMPA_NCATS)
    target_properties = {
        'qed': 1.0,
        'hERG': 0.0,          # In config
        'Caco2_Wang': -5.0,   # In config
        'PAMPA_NCATS': 1.0,   # NOT in config - should auto-load
    }
    
    print("\n" + "="*70)
    print("Testing mixed config and auto-load properties...")
    print("="*70)
    
    molecules_df = generator.optimize(
        target_properties=target_properties,
        initial_population_file="training_output_moses/no_scaffold_2_moses/2000_initial/initial_population.csv",
        population_size=50,
        generations=2,
        output_dir=output_dir,
        return_dataframe=True
    )
    
    # Verify all properties appear in output
    for prop in ['hERG', 'Caco2_Wang', 'PAMPA_NCATS']:
        assert prop in molecules_df.columns, f"{prop} should be in output columns"
        assert f"{prop}_raw" in molecules_df.columns, f"{prop}_raw should be in output columns"
    
    print("✓ Successfully mixed configured and auto-loaded properties!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

