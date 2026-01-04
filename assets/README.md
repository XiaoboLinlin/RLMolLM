# RLMolLM Assets

This directory contains the organized assets needed to run RLMolLM:

## Directory Structure

```
assets/
├── models/                      # Pre-trained model weights
│   ├── moses_pretrained.pt
│   ├── zinc_pretrained.pt
│   ├── guacamol_pretrained.pt
│   └── gdb_pretrained.pt
├── configs/                     # Configuration files
│   ├── moses.json
│   ├── zinc.json
│   ├── guacamol.json
│   └── gdb.json
├── initial_populations/         # Pre-generated initial populations
│   ├── moses_2000.csv
│   ├── zinc_2000.csv
│   ├── guacamol_2000.csv
│   └── gdb_2000.csv
```

## Download Required Files

**Model weights and initial populations are NOT tracked in git due to their size.**

### Option 1: Automatic Download (Recommended)

```bash
python download_assets.py
```

This script will download all required files from our repository.

### Option 2: Manual Download

Download from: [Link to be added]

Place files in the appropriate directories:
- Model weights → `assets/models/`
- Initial populations → `assets/initial_populations/`

## File Sizes

- **Model weights**: ~416MB each (PyTorch .pt files)
- **Initial populations**: ~100KB each (CSV files)
- **Config files**: ~2KB each (JSON files) ✅ Tracked in git

## Usage

After downloading, use the convenience functions to access these files:

```python
from rlmollm import get_model_path, get_config_path, get_population_path

# These will automatically point to the assets directory
model = get_model_path("moses")        # assets/models/moses_pretrained.pt
config = get_config_path("moses")      # assets/configs/moses.json
population = get_population_path("moses")  # assets/initial_populations/moses_2000.csv
```

## Supported Datasets

- **moses**: MOSES dataset (molecular sets)
- **zinc**: ZINC dataset
- **guacamol**: GuacaMol benchmark
- **gdb**: GDB-13 dataset

