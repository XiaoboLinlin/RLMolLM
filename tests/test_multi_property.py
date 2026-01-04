"""Test multi-property optimization with QED, LogP, and SA score."""

import pytest
import pandas as pd
from rdkit import Chem
import os
import re

@pytest.mark.parametrize("model_type", ["lm", "lm_ppo", "alm"])
def test_multi_property_optimization_model_types(generator, model_type):
    """Test optimization with multiple properties using different model types."""
    # Create output directory structure like training outputs
    output_dir = f"test_output_pytest/multi_property/{model_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    molecules = generator.optimize(
        target_properties={
            'qed': 1.0,       # Maximize drug-likeness
            'logp': 2.5,      # Target LogP of 2.5
            'sa': 1.0         # Target SA score of 1.0 (easier to synthesize)
        },
        model_type=model_type,  # Test different model types
        initial_population_file="training_output_moses/no_scaffold_2_moses/2000_initial/initial_population.csv",  # MLM needs initial population
        population_size=200,
        generations=5,
        output_dir=output_dir,  # Save output with proper structure
        return_dataframe=True  # Get DataFrame to check properties
    )
    
    # Check output
    assert isinstance(molecules, pd.DataFrame)
    assert len(molecules) == 200
    
    # Check all SMILES are valid
    for smiles in molecules['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Invalid SMILES: {smiles}"
    
    # Parse log file to verify fitness improvement across generations
    log_file = f"{output_dir}/run.log"
    assert os.path.exists(log_file), f"Log file not found: {log_file}"
    
    with open(log_file, 'r') as f:
        log_content = f.read()
    
    # Extract fitness scores from each generation
    # Pattern: [N/5] ... followed by line with fitness: X.XXXX
    # Using DOTALL to match across newlines
    pattern = r'\[(\d+)/5\].*?fitness:\s+([\d.]+)'
    matches = re.findall(pattern, log_content, re.DOTALL)
    
    assert len(matches) >= 2, f"Not enough generations found in log (found {len(matches)})"
    
    generations = []
    fitness_scores = []
    for gen_num, fitness in matches:
        generations.append(int(gen_num))
        fitness_scores.append(float(fitness))
    
    print(f"\n✓ Model type: {model_type}")
    print(f"✓ Generated {len(molecules)} valid molecules")
    print(f"✓ Fitness progression across {len(fitness_scores)} generations:")
    for gen, fitness in zip(generations, fitness_scores):
        print(f"    Generation {gen}: fitness = {fitness:.4f}")
    
    # Verify fitness is improving (allow some fluctuation but overall trend should be up)
    initial_fitness = fitness_scores[0]
    final_fitness = fitness_scores[-1]
    
    print(f"\n  Initial fitness: {initial_fitness:.4f}")
    print(f"  Final fitness:   {final_fitness:.4f}")
    print(f"  Improvement:     {(final_fitness - initial_fitness):.4f} ({((final_fitness/initial_fitness - 1) * 100):.1f}%)")
    
    # Assert that fitness improved from generation 0 to final generation
    assert final_fitness > initial_fitness, \
        f"Fitness did not improve: initial={initial_fitness:.4f}, final={final_fitness:.4f}"
    
    # Check that there's meaningful improvement (at least 5%)
    improvement_pct = (final_fitness / initial_fitness - 1) * 100
    assert improvement_pct >= 5.0, \
        f"Fitness improvement too small: {improvement_pct:.1f}% (expected >= 5%)"
    
    print(f"\n✓ Fitness improved significantly across generations")
    print(f"✓ Multi-property optimization completed successfully for {model_type}")


def test_multi_property_optimization(generator):
    """Test optimization with multiple properties simultaneously."""
    # Create output directory structure like training outputs
    output_dir = "test_output_pytest/multi_property/lm"
    os.makedirs(output_dir, exist_ok=True)
    
    molecules = generator.optimize(
        target_properties={
            'qed': 1.0,       # Maximize drug-likeness
            'logp': 2.5,      # Target LogP of 2.5
            'sa': 1.0         # Target SA score of 1.0 (easier to synthesize)
        },
        initial_population_file="training_output_moses/no_scaffold_2_moses/2000_initial/initial_population.csv",  # MLM needs initial population
        population_size=200,
        generations=5,
        output_dir=output_dir,  # Save output with proper structure
        return_dataframe=True  # Get DataFrame to check properties
    )
    
    # Check output
    assert isinstance(molecules, pd.DataFrame)
    assert len(molecules) == 200
    
    # Check all SMILES are valid
    for smiles in molecules['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Invalid SMILES: {smiles}"
    
    # Parse log file to verify fitness improvement across generations
    log_file = f"{output_dir}/run.log"
    assert os.path.exists(log_file), f"Log file not found: {log_file}"
    
    with open(log_file, 'r') as f:
        log_content = f.read()
    
    # Extract fitness scores from each generation
    # Pattern: [N/5] ... followed by line with fitness: X.XXXX
    # Using DOTALL to match across newlines
    pattern = r'\[(\d+)/5\].*?fitness:\s+([\d.]+)'
    matches = re.findall(pattern, log_content, re.DOTALL)
    
    assert len(matches) >= 2, f"Not enough generations found in log (found {len(matches)})"
    
    generations = []
    fitness_scores = []
    for gen_num, fitness in matches:
        generations.append(int(gen_num))
        fitness_scores.append(float(fitness))
    
    print(f"✓ Generated {len(molecules)} valid molecules")
    print(f"✓ Fitness progression across {len(fitness_scores)} generations:")
    for gen, fitness in zip(generations, fitness_scores):
        print(f"    Generation {gen}: fitness = {fitness:.4f}")
    
    # Verify fitness is improving (allow some fluctuation but overall trend should be up)
    initial_fitness = fitness_scores[0]
    final_fitness = fitness_scores[-1]
    
    print(f"\n  Initial fitness: {initial_fitness:.4f}")
    print(f"  Final fitness:   {final_fitness:.4f}")
    print(f"  Improvement:     {(final_fitness - initial_fitness):.4f} ({((final_fitness/initial_fitness - 1) * 100):.1f}%)")
    
    # Assert that fitness improved from generation 0 to final generation
    assert final_fitness > initial_fitness, \
        f"Fitness did not improve: initial={initial_fitness:.4f}, final={final_fitness:.4f}"
    
    # Check that there's meaningful improvement (at least 5%)
    improvement_pct = (final_fitness / initial_fitness - 1) * 100
    assert improvement_pct >= 5.0, \
        f"Fitness improvement too small: {improvement_pct:.1f}% (expected >= 5%)"
    
    print(f"\n✓ Fitness improved significantly across generations")
    print("✓ Multi-property optimization completed successfully")

