"""Test list and DataFrame output formats."""

import pytest
import pandas as pd
from rdkit import Chem
from rlmollm.config import get_population_path

def test_list_output(generator):
    """Test default list output."""
    molecules = generator.optimize(
        target_properties={'qed': 1.0},
        initial_population_file=get_population_path("moses"),
        population_size=32,
        generations=2
    )
    
    assert isinstance(molecules, list)
    assert len(molecules) == 32
    assert all(isinstance(s, str) for s in molecules)
    
    # All valid SMILES
    assert all(Chem.MolFromSmiles(s) is not None for s in molecules)
    print(f"✓ List output: {len(molecules)} molecules")


def test_dataframe_output(generator):
    """Test DataFrame output with return_dataframe=True."""
    df = generator.optimize(
        target_properties={'qed': 1.0, 'logp': 2.5},
        initial_population_file=get_population_path("moses"),
        population_size=32,
        generations=2,
        return_dataframe=True
    )
    
    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 32
    
    # Check required columns exist
    assert 'smiles' in df.columns
    assert 'fitness' in df.columns
    
    # All SMILES are valid
    for smiles in df['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
    
    # Properties are numeric
    assert df['fitness'].dtype in ['float64', 'float32']
    
    print(f"✓ DataFrame output: {len(df)} rows, {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")


