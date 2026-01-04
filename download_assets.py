#!/usr/bin/env python3
"""
RLMolLM Assets Download Script

This script downloads pre-trained models and initial populations from Hugging Face.
Files are organized into the assets/ directory structure.

Usage:
    python download_assets.py [--dataset moses|zinc|guacamol|gdb|all]

Requirements:
    pip install huggingface_hub

Authentication:
    Set the HF_TOKEN environment variable:
    export HF_TOKEN="your_hugging_face_token"
    
    Or the script will prompt for it interactively.
"""

import os
import sys
import argparse
from pathlib import Path


def install_requirements():
    """Install required packages if not available."""
    try:
        from huggingface_hub import hf_hub_download
        print("✅ huggingface_hub is already installed")
        return True
    except ImportError:
        print("📦 Installing huggingface_hub...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
            print("✅ huggingface_hub installed successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to install huggingface_hub: {e}")
            print("Please install manually: pip install huggingface_hub")
            return False


def get_auth_token():
    """Get authentication token from environment or user input."""
    # Try environment variable first
    token = os.environ.get("HF_TOKEN")
    if token:
        print("✅ Using token from HF_TOKEN environment variable")
        return token
    
    # Check if huggingface-cli is logged in
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            print("✅ Using token from huggingface-cli login")
            return token
    except:
        pass
    
    # Prompt user
    print("\n⚠️  No authentication token found.")
    print("You can:")
    print("  1. Set HF_TOKEN environment variable: export HF_TOKEN='your_token'")
    print("  2. Login via: huggingface-cli login")
    print("  3. Enter token now (input will be hidden)")
    
    try:
        import getpass
        token = getpass.getpass("Enter Hugging Face token (or press Enter to skip): ")
        if token.strip():
            return token.strip()
    except:
        pass
    
    print("⚠️  No token provided. Attempting public download (may fail for private repos)...")
    return None


def download_assets(datasets=["moses"], repo_id="scofieldlinlin/rlmollm-assets", token=None):
    """Download model weights and initial populations from Hugging Face."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("❌ huggingface_hub not available. Please install it first.")
        return False
    
    # Get the project root directory
    project_root = Path(__file__).parent
    assets_dir = project_root / "assets"
    
    # Create directory structure
    (assets_dir / "models").mkdir(parents=True, exist_ok=True)
    (assets_dir / "initial_populations").mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Assets directory: {assets_dir}")
    print(f"🔗 Downloading from repository: {repo_id}")
    
    # Define files to download for each dataset
    files_map = {
        "moses": [
            ("models/moses_pretrained.pt", "assets/models/moses_pretrained.pt"),
            ("initial_populations/moses_2000.csv", "assets/initial_populations/moses_2000.csv"),
        ],
        "zinc": [
            ("models/zinc_pretrained.pt", "assets/models/zinc_pretrained.pt"),
            ("initial_populations/zinc_2000.csv", "assets/initial_populations/zinc_2000.csv"),
        ],
        "guacamol": [
            ("models/guacamol_pretrained.pt", "assets/models/guacamol_pretrained.pt"),
            ("initial_populations/guacamol_2000.csv", "assets/initial_populations/guacamol_2000.csv"),
        ],
        "gdb": [
            ("models/gdb_pretrained.pt", "assets/models/gdb_pretrained.pt"),
            ("initial_populations/gdb_2000.csv", "assets/initial_populations/gdb_2000.csv"),
        ],
    }
    
    # Download files for each dataset
    success_count = 0
    fail_count = 0
    
    for dataset in datasets:
        if dataset not in files_map:
            print(f"⚠️  Unknown dataset: {dataset}")
            continue
        
        print(f"\n📥 Downloading {dataset} assets...")
        
        for remote_path, local_path in files_map[dataset]:
            local_file = project_root / local_path
            
            # Skip if file already exists
            if local_file.exists():
                file_size = local_file.stat().st_size
                print(f"⏭️  {local_file.name} already exists ({file_size:,} bytes), skipping...")
                success_count += 1
                continue
            
            try:
                print(f"📥 Downloading {remote_path}...")
                
                # Download to cache first, then copy to destination
                cached_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=remote_path,
                    token=token,
                    repo_type="model"
                )
                
                # Copy from cache to local destination
                import shutil
                local_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cached_path, local_file)
                
                # Verify file was downloaded
                if local_file.exists():
                    file_size = local_file.stat().st_size
                    print(f"✅ {local_file.name} downloaded ({file_size:,} bytes)")
                    success_count += 1
                else:
                    print(f"❌ {local_file.name} download failed (file not found)")
                    fail_count += 1
                    
            except Exception as e:
                print(f"❌ Failed to download {remote_path}: {e}")
                fail_count += 1
    
    print(f"\n📊 Download Summary: {success_count} succeeded, {fail_count} failed")
    return fail_count == 0


def verify_assets(datasets=["moses"]):
    """Verify that all required files are present and have reasonable sizes."""
    project_root = Path(__file__).parent
    
    # Expected file sizes (minimum)
    expected_files = {
        "moses": [
            ("assets/models/moses_pretrained.pt", 400_000_000),  # ~400MB
            ("assets/initial_populations/moses_2000.csv", 1_000_000),  # ~1MB
        ],
        "zinc": [
            ("assets/models/zinc_pretrained.pt", 400_000_000),
            ("assets/initial_populations/zinc_2000.csv", 1_000_000),
        ],
        "guacamol": [
            ("assets/models/guacamol_pretrained.pt", 400_000_000),
            ("assets/initial_populations/guacamol_2000.csv", 1_000_000),
        ],
        "gdb": [
            ("assets/models/gdb_pretrained.pt", 400_000_000),
            ("assets/initial_populations/gdb_2000.csv", 1_000_000),
        ],
    }
    
    print("\n🔍 Verifying downloads...")
    
    all_good = True
    for dataset in datasets:
        if dataset not in expected_files:
            continue
        
        print(f"\n{dataset}:")
        for rel_path, min_size in expected_files[dataset]:
            file_path = project_root / rel_path
            
            if not file_path.exists():
                print(f"  ❌ {file_path.name} not found")
                all_good = False
                continue
            
            file_size = file_path.stat().st_size
            if file_size < min_size:
                print(f"  ❌ {file_path.name} too small ({file_size:,} bytes, expected >{min_size:,})")
                all_good = False
                continue
            
            print(f"  ✅ {file_path.name} ({file_size:,} bytes)")
    
    return all_good


def main():
    """Main function to download and set up assets."""
    parser = argparse.ArgumentParser(description="Download RLMolLM assets from Hugging Face")
    parser.add_argument(
        "--dataset",
        choices=["moses", "zinc", "guacamol", "gdb", "all"],
        default="all",
        help="Which dataset to download (default: all)"
    )
    parser.add_argument(
        "--repo-id",
        default="scofieldlinlin/rlmollm-assets",
        help="Hugging Face repository ID"
    )
    args = parser.parse_args()
    
    print("🚀 RLMolLM Assets Download Script")
    print("=" * 50)
    
    # Determine which datasets to download
    if args.dataset == "all":
        datasets = ["moses", "zinc", "guacamol", "gdb"]
    else:
        datasets = [args.dataset]
    
    print(f"📦 Datasets to download: {', '.join(datasets)}")
    
    # Step 1: Install requirements
    if not install_requirements():
        return 1
    
    # Step 2: Get authentication token
    token = get_auth_token()
    
    # Step 3: Download assets
    if not download_assets(datasets=datasets, repo_id=args.repo_id, token=token):
        print("\n⚠️  Some downloads failed. You can:")
        print("  1. Check your internet connection")
        print("  2. Verify your Hugging Face token has access to the repo")
        print("  3. Try downloading specific datasets with --dataset moses")
        # Don't return error - partial success is ok
    
    # Step 4: Verify downloads
    if not verify_assets(datasets=datasets):
        print("\n⚠️  Some files may be missing or corrupted.")
        print("Try running the script again.")
        return 1
    
    print("\n🎉 Assets download completed successfully!")
    print("📁 Files saved to: assets/")
    print("\n🚀 You can now use RLMolLM:")
    print("   from rlmollm import RLMolLMGenerator, get_model_path")
    print("   generator = RLMolLMGenerator(")
    print("       checkpoint_path=get_model_path('moses'),")
    print("       config_path=get_config_path('moses')")
    print("   )")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

