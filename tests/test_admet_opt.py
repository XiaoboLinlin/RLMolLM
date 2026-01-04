"""Test ADMET property optimization with auto-loading and raw values.

This comprehensive test verifies:
1. Properties NOT in config are automatically loaded
2. Both normalized and raw values are returned
3. Properties improve over generations
4. All SMILES are valid
"""

import pytest
import pandas as pd
from rdkit import Chem
import os
import re


def test_auto_load_property_not_in_config(generator):
    """Comprehensive test: auto-load, raw values, and property improvement."""
    output_dir = "tests/test_output/comprehensive_admet"
    os.makedirs(output_dir, exist_ok=True)
    
    # PAMPA_NCATS and AMES are NOT in config - should auto-load!
    target_properties = {
        'qed': 1.0,                    # Maximize drug-likeness
        'sa': 1.0,                     # Easier to synthesize
        'PAMPA_NCATS': 1.0,            # NOT in config - auto-load!
        'AMES': 0.0,                   # NOT in config - auto-load!
    }
    
    print("\n" + "="*70)
    print("Testing: Auto-load + Raw Values + Property Improvement")
    print("="*70)
    
    molecules_df = generator.optimize(
        target_properties=target_properties,
        initial_population_file="training_output_moses/no_scaffold_2_moses/2000_initial/initial_population.csv",
        population_size=1000,
        generations=3,
        output_dir=output_dir,
        return_dataframe=True
    )
    
    # ===== Test 1: Auto-Load Works =====
    print("\n✓ Test 1: Auto-load properties NOT in config")
    assert 'PAMPA_NCATS' in molecules_df.columns, "PAMPA_NCATS should be auto-loaded"
    assert 'PAMPA_NCATS_raw' in molecules_df.columns, "PAMPA_NCATS_raw should be auto-loaded"
    assert 'AMES' in molecules_df.columns, "AMES should be auto-loaded"
    assert 'AMES_raw' in molecules_df.columns, "AMES_raw should be auto-loaded"
    print(f"    ✓ PAMPA_NCATS auto-loaded from property_configs.py")
    print(f"    ✓ AMES auto-loaded from property_configs.py")
    
    # ===== Test 2: Raw Values Present =====
    print("\n✓ Test 2: Raw and normalized values present")
    # Use user-facing property names (consistent with input)
    expected_props = ['qed', 'sa', 'PAMPA_NCATS', 'AMES']
    for prop in expected_props:
        assert prop in molecules_df.columns, f"Missing normalized: {prop}"
        assert f'{prop}_raw' in molecules_df.columns, f"Missing raw: {prop}_raw"
    print(f"    ✓ All properties have both normalized and raw values")
    
    # Verify normalized values are in [0, 1]
    for prop in expected_props:
        assert molecules_df[prop].min() >= 0.0, f"{prop} normalized should be >= 0"
        assert molecules_df[prop].max() <= 1.0, f"{prop} normalized should be <= 1"
    print(f"    ✓ All normalized values in valid range [0, 1]")
    
    # Show value ranges
    print(f"\n  Property Ranges:")
    print(f"    PAMPA_NCATS: norm [{molecules_df['PAMPA_NCATS'].min():.3f}, {molecules_df['PAMPA_NCATS'].max():.3f}], "
          f"raw [{molecules_df['PAMPA_NCATS_raw'].min():.3f}, {molecules_df['PAMPA_NCATS_raw'].max():.3f}]")
    print(f"    AMES:        norm [{molecules_df['AMES'].min():.3f}, {molecules_df['AMES'].max():.3f}], "
          f"raw [{molecules_df['AMES_raw'].min():.3f}, {molecules_df['AMES_raw'].max():.3f}]")
    
    # ===== Test 3: Property Improvement =====
    print("\n✓ Test 3: Properties improve over generations")
    
    # Check log file for fitness improvement
    log_file = f"{output_dir}/run.log"
    assert os.path.exists(log_file), f"Log file not found: {log_file}"
    
    with open(log_file, 'r') as f:
        log_content = f.read()
    
    # Extract fitness scores from each generation
    pattern = r'\[(\d+)/3\].*?fitness:\s+([\d.]+)'
    matches = re.findall(pattern, log_content, re.DOTALL)
    assert len(matches) >= 2, f"Not enough generations found in log (found {len(matches)})"
    
    fitness_scores = [float(fitness) for gen_num, fitness in matches]
    initial_fitness = fitness_scores[0]
    final_fitness = fitness_scores[-1]
    
    print(f"    Initial fitness: {initial_fitness:.4f}")
    print(f"    Final fitness:   {final_fitness:.4f}")
    print(f"    Improvement:     {(final_fitness - initial_fitness):.4f} "
          f"({((final_fitness/initial_fitness - 1) * 100):.1f}%)")
    
    assert final_fitness > initial_fitness, \
        f"Fitness did not improve: initial={initial_fitness:.4f}, final={final_fitness:.4f}"
    print(f"    ✓ Fitness improved over generations")
    
    # ===== Test 4: Valid SMILES =====
    print("\n✓ Test 4: All SMILES are valid")
    invalid_count = sum(1 for smiles in molecules_df['smiles'] if Chem.MolFromSmiles(smiles) is None)
    assert invalid_count == 0, f"Found {invalid_count} invalid SMILES"
    print(f"    ✓ All {len(molecules_df)} SMILES are valid")
    
    # ===== Test 5: CSV Output =====
    print("\n✓ Test 5: CSV output contains all columns")
    csv_file = f"{output_dir}/final_population.csv"
    df_csv = pd.read_csv(csv_file)
    
    for prop in ['PAMPA_NCATS', 'PAMPA_NCATS_raw', 'AMES', 'AMES_raw']:
        assert prop in df_csv.columns, f"CSV missing column: {prop}"
    print(f"    ✓ CSV has all raw and normalized columns")
    
    # ===== Final Summary =====
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("✓ Auto-loading: Properties not in config were loaded automatically")
    print("✓ Raw values: Both raw and normalized values present")
    print("✓ Improvement: Fitness improved over generations")
    print("✓ Validity: All molecules are valid")
    print("✓ Output: CSV contains all expected columns")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Allow running directly for quick testing
    from rlmollm import RLMolLMGenerator
    
    gen = RLMolLMGenerator(
        checkpoint_path="model_weights_moses/pretrain_0p5/latest_model_run_generator_0.pt",
        config_path="config/no_scaffold_2_moses.json",
        verbose=True
    )
    
    print("Running comprehensive ADMET test...")
    test_auto_load_property_not_in_config(gen)
    print("\n✅ Test completed successfully!")
