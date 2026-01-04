# RLMolLM: Reinforcement Learning-Enhanced Language Model Framework for Inverse Molecular Design

A Python library for AI-driven molecular generation and optimization using reinforcement learning. Generate drug-like molecules optimized for multiple properties simultaneously.

## Features

- 🎯 **Multi-property optimization** - Optimize QED, LogP, SA, and 49+ ADMET properties
- 🔬 **Scaffold-based generation** - Constrain molecules to specific scaffolds
- 🚀 **Simple API** - Just import and use, no scripts needed
- 📊 **Built-in analysis** - Returns DataFrames with all calculated properties
- 🧪 **Well-tested** - Comprehensive test suite included

## Installation

### Recommended: Using Conda

```bash
git clone https://github.com/XiaoboLinlin/RLMolLM.git
cd RLMolLM

# Create Python 3.11+ environment
conda create -n rlmollm-env python=3.11
conda activate rlmollm-env

# Install package
pip install -e .
```

### Alternative: Pip-only Installation

If you prefer not to use conda or don't have it installed:

```bash
git clone https://github.com/XiaoboLinlin/RLMolLM.git
cd RLMolLM
bash install_pip_only.sh
source rlmollm_env/bin/activate
```

### Download Pre-trained Models

Model weights (~417MB) and initial populations are required:

```bash
python download_assets.py --dataset moses
```

## Quick Start

### Basic Usage

```python
from rlmollm import RLMolLMGenerator, get_model_path, get_config_path, get_population_path

# Initialize generator (using convenience functions for paths)
generator = RLMolLMGenerator(
    checkpoint_path=get_model_path("moses"),
    config_path=get_config_path("moses")
)

# Optimize molecules for drug-likeness
molecules_df = generator.optimize(
    target_properties={'qed': 1.0},  # Maximize QED
    initial_population_file=get_population_path("moses"),
    population_size=200,
    generations=5,
    output_dir="output/my_optimization",
    return_dataframe=True
)

# Results are in a DataFrame
print(molecules_df[['smiles', 'qed', 'fitness']].head())
```

**Alternative: Use custom paths**

```python
# You can still provide custom paths
generator = RLMolLMGenerator(
    checkpoint_path="path/to/your/model.pt",
    config_path="path/to/your/config.json"
)
```

### Multi-Property Optimization

```python
from rlmollm import RLMolLMGenerator, get_model_path, get_config_path, get_population_path

generator = RLMolLMGenerator(
    checkpoint_path=get_model_path("moses"),
    config_path=get_config_path("moses")
)

# Optimize multiple properties simultaneously
molecules_df = generator.optimize(
    target_properties={
        'qed': 1.0,        # Maximize drug-likeness
        'logp': 2.5,       # Target LogP of 2.5
        'sa': 1.0,         # Easier to synthesize (Synthetic Accessibility)
    },
    initial_population_file=get_population_path("moses"),
    population_size=200,
    generations=5,
    output_dir="output/multi_property",
    return_dataframe=True
)
```

### Scaffold-Based Generation

```python
# Generate molecules constrained to a benzene scaffold
# Use '#' markers to indicate attachment points
molecules_df = generator.optimize(
    target_properties={
        'qed': 1.0,
        'logp': 2.5,
    },
    use_scaffold=True,
    scaffold_smiles="#c1cc(#)ccc1#",  # Benzene with 3 attachment points
    population_size=100,
    generations=5,
    output_dir="output/benzene_scaffold",
    return_dataframe=True
)

# All molecules will contain the benzene core
```

### ADMET Property Optimization

Optimize for 49+ ADMET properties - they're automatically loaded even if not in your config!

```python
molecules_df = generator.optimize(
    target_properties={
        'qed': 1.0,
        'PAMPA_NCATS': 1.0,         # Maximize permeability
        'hERG': 0.0,                # Minimize cardiac toxicity
        'Caco2_Wang': 1.0,          # Maximize Caco-2 permeability
        'AMES': 0.0,                # Minimize mutagenicity
    },
    initial_population_file="training_output_moses/no_scaffold_2_moses/2000_initial/initial_population.csv",
    population_size=200,
    generations=5,
    output_dir="output/admet",
    return_dataframe=True
)

# Both raw and normalized values are included
print(molecules_df[['smiles', 'PAMPA_NCATS', 'PAMPA_NCATS_raw', 'hERG', 'hERG_raw']].head())
```

### Model Types

Choose different training strategies:

```python
# Language Model with PPO reinforcement learning
molecules_df = generator.optimize(
    target_properties={'qed': 1.0},
    model_type='lm_ppo',  # Options: 'lm', 'lm_ppo', 'alm', 'alm_ppo'
    initial_population_file="...",
    population_size=200,
    generations=5,
    output_dir="output/lm_ppo",
    return_dataframe=True
)
```

## Available Properties

### Basic Properties
- `qed` - Drug-likeness (Quantitative Estimate of Drug-likeness)
- `logp` - Lipophilicity
- `sa` - Synthetic Accessibility (1 = easy to synthesize, 10 = hard)
- `tpsa` - Topological Polar Surface Area

### ADMET Properties (49+ available)

**Absorption & Permeability:**
- `HIA_Hou`, `Caco2_Wang`, `PAMPA_NCATS`, `Bioavailability_Ma`

**Metabolism (CYP):**
- `CYP3A4_Substrate_CarbonMangels`, `CYP2D6_Substrate_CarbonMangels`, `CYP1A2_Veith`, etc.

**Distribution:**
- `BBB_Martins` (Blood-brain barrier), `PPBR_AZ`, `VDss_Lombardo`

**Toxicity:**
- `hERG` (Cardiac toxicity), `AMES` (Mutagenicity), `DILI` (Liver injury), `LD50_Zhu`

**Physicochemical:**
- `Solubility_AqSolDB`, `Lipophilicity_AstraZeneca`, `HydrationFreeEnergy_FreeSolv`

See full list in `rlmollm/scoring/property_configs.py`.

## Output

The generator returns a pandas DataFrame with:
- `smiles` - Generated molecule SMILES
- Property columns (both normalized and `_raw` values)
- `fitness` - Overall fitness score (harmonic mean of properties)

CSV files are also saved to the output directory:
- `final_population.csv` - Optimized molecules
- `initial_population_properties.csv` - Starting molecules with properties
- `run.log` - Optimization log

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_multi_property.py -v

# Test ADMET optimization
pytest tests/test_admet_opt.py -v

# Test scaffold-based generation
pytest tests/test_scaffold.py -v
```

## Requirements

- Python 3.11+
- PyTorch 2.4.0+
- Transformers 4.50.0+
- RDKit 2024.3.3+
- ADMET-AI 1.3.1+

Full list in `requirements.txt`.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{lin2025rlmollm,
  title={RLMolLM: Reinforcement Learning-Enhanced Language Model Framework for Inverse Molecular Design},
  author={Lin, Xiaobo and Bhowmik, Debsindhu and Kearney, Logan T and Naskar, Amit K},
  journal={Journal of Chemical Information and Modeling},
  volume={65},
  number={22},
  pages={12292--12304},
  year={2025},
  publisher={ACS Publications}
}
```
```

## License

[Your License Here]

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- See test files for more usage examples
