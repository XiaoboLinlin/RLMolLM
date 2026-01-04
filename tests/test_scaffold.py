"""Test scaffold-based multi-property optimization."""

import pytest
import pandas as pd
from rdkit import Chem
import os

# Define scaffold test cases: (name, scaffold_with_markers, core_without_markers)
# These scaffolds are from config/scaffold_examples/ and are validated to work
SCAFFOLD_TEST_CASES = [
    ("benzene", "#c1cc(#)ccc1#", "c1ccccc1"),
    ("dihydropyridine", "#C1=C(#)NC(#)=C(#)C1#", "C1=CNC=CC1"),
    ("benzothiophene", "c1c(#)cc2sc(#)c(#)c2c1#", "c1ccc2sccc2c1"),
]


@pytest.mark.parametrize("scaffold_name,scaffold_smiles,core_smiles", SCAFFOLD_TEST_CASES)
def test_scaffold_multi_property_optimization(generator, scaffold_name, scaffold_smiles, core_smiles):
    """Test optimization with different scaffold constraints and multiple properties."""
    
    # Create output directory in tests folder
    output_dir = f"tests/test_output/scaffold/{scaffold_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Note: Scaffold-based generation creates its own initial population from the scaffold
    # No need to provide initial_population_file
    molecules = generator.optimize(
        target_properties={
            'qed': 1.0,       # Maximize drug-likeness
            'logp': 2.5,      # Target LogP of 2.5
            'sa_score': 1.0   # Target SA score of 1.0
        },
        use_scaffold=True,
        scaffold_smiles=scaffold_smiles,
        population_size=100,  # Smaller population for faster testing
        generations=3,
        output_dir=output_dir,  # Need output_dir for scaffold-based generation
        return_dataframe=True  # Get DataFrame to check properties
    )
    
    # Check output
    assert isinstance(molecules, pd.DataFrame)
    assert len(molecules) == 100
    
    # Get core structure for substructure matching (without # markers)
    scaffold_mol = Chem.MolFromSmiles(core_smiles)
    assert scaffold_mol is not None, f"Invalid core SMILES: {core_smiles}"
    
    # Check all molecules are valid and contain the scaffold core
    for smiles in molecules['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Invalid SMILES: {smiles}"
        
        # Check molecule contains scaffold core as substructure
        assert mol.HasSubstructMatch(scaffold_mol), \
            f"Molecule {smiles} does not contain {scaffold_name} core ({core_smiles})"
    
    # Check that properties are being optimized toward targets
    avg_fitness = molecules['fitness'].mean()
    
    print(f"\n✓ Scaffold: {scaffold_name}")
    print(f"  Template: {scaffold_smiles}")
    print(f"  Core: {core_smiles}")
    print(f"✓ Generated {len(molecules)} valid molecules")
    print(f"✓ All molecules contain {scaffold_name} core")
    print(f"  Average fitness score: {avg_fitness:.3f}")
    
    # Fitness score should be reasonable
    assert avg_fitness > 0, f"Fitness score too low: {avg_fitness:.3f}"
    
    print(f"✓ {scaffold_name.capitalize()} scaffold optimization completed successfully")


if __name__ == "__main__":
    # Allow running directly with: python test_scaffold.py
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))


