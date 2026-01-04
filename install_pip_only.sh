#!/bin/bash
# RLMolLM Installation Script (pip-based)
# This script installs RLMolLM using pip only

set -e  # Exit on error

echo "========================================================================"
echo "RLMolLM Installation Script"
echo "========================================================================"
echo ""

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: pyproject.toml not found!${NC}"
    echo "Please run this script from the RLMolLM root directory."
    exit 1
fi

echo -e "${GREEN}✓${NC} Found RLMolLM package"
echo ""

# Check Python version
PYTHON_CMD="python"
if ! command -v python &> /dev/null; then
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        echo -e "${RED}Error: Python not found!${NC}"
        exit 1
    fi
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Check if Python version is >= 3.11
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]); then
    echo -e "${YELLOW}Warning: Python $PYTHON_VERSION found, but Python 3.11+ is required${NC}"
    
    # Check if conda is available with a suitable Python
    if command -v conda &> /dev/null; then
        echo "Checking for conda Python 3.11+..."
        if [ -f "$HOME/miniconda3/bin/python" ]; then
            CONDA_VERSION=$($HOME/miniconda3/bin/python --version 2>&1 | awk '{print $2}')
            CONDA_MAJOR=$(echo $CONDA_VERSION | cut -d. -f1)
            CONDA_MINOR=$(echo $CONDA_VERSION | cut -d. -f2)
            if [ "$CONDA_MAJOR" -ge 3 ] && [ "$CONDA_MINOR" -ge 11 ]; then
                echo -e "${GREEN}✓${NC} Found conda Python $CONDA_VERSION"
                PYTHON_CMD="$HOME/miniconda3/bin/python"
            fi
        fi
    fi
    
    # Final check
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]); then
        echo -e "${RED}Error: Python 3.11+ is required. Found: $PYTHON_VERSION${NC}"
        echo "Please install Python 3.11 or higher."
        exit 1
    fi
fi

echo "Using Python: $PYTHON_CMD ($PYTHON_VERSION)"

# Check if virtual environment exists
if [ -d "venv" ] || [ -d "rlmollm_env" ]; then
    echo -e "${YELLOW}Warning: Virtual environment already exists${NC}"
    echo "Using existing environment..."
    ENV_DIR="venv"
    if [ -d "rlmollm_env" ]; then
        ENV_DIR="rlmollm_env"
    fi
else
    echo ""
    echo "Creating virtual environment..."
    ENV_DIR="rlmollm_env"
    
    # Try to create venv
    if $PYTHON_CMD -m venv $ENV_DIR --without-pip 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Virtual environment created"
        
        # Install pip manually
        echo "Installing pip..."
        curl -sSf https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        $ENV_DIR/bin/python get-pip.py
        rm get-pip.py
        echo -e "${GREEN}✓${NC} Pip installed"
    else
        # If venv creation fails, try with pip
        echo "Trying alternative method..."
        $PYTHON_CMD -m venv $ENV_DIR
        echo -e "${GREEN}✓${NC} Virtual environment created"
    fi
fi

echo ""
echo "Activating virtual environment..."
source $ENV_DIR/bin/activate

echo -e "${GREEN}✓${NC} Environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

echo ""
echo "Installing RLMolLM package and dependencies..."
echo "This may take several minutes..."
echo ""

# Install the package
pip install -e .

echo ""
echo "========================================================================"
echo -e "${GREEN}✓ Installation Complete!${NC}"
echo "========================================================================"
echo ""
echo "To activate the environment in the future, run:"
echo "  source $ENV_DIR/bin/activate"
echo ""
echo "To verify installation, run:"
echo '  python -c "from rlmollm import RLMolLMGenerator; print('"'"'✓ RLMolLM working!'"'"')"'
echo ""
echo "To deactivate the environment, run:"
echo "  deactivate"
echo ""

# Run quick verification test
echo "Running verification test..."
python -c "
import sys
try:
    from rlmollm import RLMolLMGenerator
    from rlmollm.models.gan import Gan
    from rlmollm.token_splits import pretokenizer_dict
    print('✓ All imports successful!')
    print('✓ RLMolLM is ready to use!')
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
" && echo -e "${GREEN}✓ Verification passed!${NC}" || echo -e "${RED}✗ Verification failed!${NC}"

echo ""
echo "For optional dependencies, you can also install:"
echo "  pip install -e \".[viz]\"   # Visualization tools"
echo "  pip install -e \".[test]\"  # Testing tools"
echo "  pip install -e \".[all]\"   # All optional dependencies"

