"""
Dataset loader for LSTM highlight detection pipeline.
Handles downloading and preparing:
1. Person Segmentation Dataset (Kaggle) - for pre-training
2. Freestyle-Wrestling-Dataset (GitHub) - for fine-tuning
"""

import os
import requests
import zipfile
import tarfile
from pathlib import Path
import shutil
from tqdm import tqdm
import json

# Configuration constants
DATASET_CONFIG = {
    "kaggle_person_segmentation": {
        "name": "Person Segmentation Dataset",
        "url": "https://www.kaggle.com/datasets/balraj98/person-segmentation-dataset",
        "local_path": "datasets/kaggle_person_segmentation",
        "description": "Pre-trained segmentation model weights"
    },
    "freestyle_wrestling": {
        "name": "Freestyle-Wrestling-Dataset",
        "url": "https://github.com/your-repo/freestyle-wrestling-dataset",  # Need actual URL
        "local_path": "datasets/freestyle_wrestling",
        "description": "Fine-tuning dataset with RGB + silhouette pairs"
    }
}

def create_dataset_directories():
    """Create necessary directories for datasets"""
    for dataset_info in DATASET_CONFIG.values():
        Path(dataset_info["local_path"]).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dataset_info['local_path']}")

def download_kaggle_dataset():
    """
    Download Person Segmentation Dataset from Kaggle.
    Note: Requires kaggle CLI and authentication.
    """
    print("📥 Downloading Person Segmentation Dataset from Kaggle...")
    
    # Check if kaggle CLI is installed
    if not shutil.which("kaggle"):
        print("❌ Kaggle CLI not found. Please install it first:")
        print("   pip install kaggle")
        print("   Then authenticate with: kaggle login")
        return False
    
    dataset_path = DATASET_CONFIG["kaggle_person_segmentation"]["local_path"]
    
    try:
        # Download dataset using kaggle CLI
        os.system(f"kaggle datasets download -d balraj98/person-segmentation-dataset -p {dataset_path}")
        
        # Extract the downloaded zip file
        zip_files = list(Path(dataset_path).glob("*.zip"))
        if zip_files:
            with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
                zip_ref.extractall(dataset_path)
            # Remove the zip file
            zip_files[0].unlink()
        
        print(f"✅ Kaggle dataset downloaded to: {dataset_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading Kaggle dataset: {e}")
        return False

def download_freestyle_wrestling_dataset():
    """
    Download Freestyle-Wrestling-Dataset from GitHub.
    Note: Need to find the actual repository URL.
    """
    print("📥 Downloading Freestyle-Wrestling-Dataset from GitHub...")
    
    # TODO: Replace with actual GitHub repository URL
    github_url = "https://github.com/actual-repo/freestyle-wrestling-dataset"
    dataset_path = DATASET_CONFIG["freestyle_wrestling"]["local_path"]
    
    try:
        # For now, create a placeholder structure
        # In reality, you'd clone the repo or download the release
        placeholder_structure = {
            "rgb_videos": [],
            "silhouette_videos": [],
            "annotations": [],
            "README.md": "Placeholder for Freestyle-Wrestling-Dataset"
        }
        
        # Create placeholder files
        for key, value in placeholder_structure.items():
            if isinstance(value, list):
                Path(dataset_path, key).mkdir(exist_ok=True)
            else:
                with open(Path(dataset_path, key), 'w') as f:
                    f.write(value)
        
        print(f"⚠️  Placeholder created for: {dataset_path}")
        print("   Please manually download the Freestyle-Wrestling-Dataset")
        print(f"   Expected URL: {github_url}")
        return False
        
    except Exception as e:
        print(f"❌ Error setting up Freestyle-Wrestling-Dataset: {e}")
        return False

def verify_dataset_structure():
    """Verify that datasets are properly structured"""
    print("🔍 Verifying dataset structure...")
    
    for dataset_name, dataset_info in DATASET_CONFIG.items():
        path = Path(dataset_info["local_path"])
        if path.exists():
            print(f"✅ {dataset_name}: {path}")
            # List contents
            contents = list(path.rglob("*"))[:5]  # Show first 5 items
            for item in contents:
                print(f"   - {item.name}")
            if len(list(path.rglob("*"))) > 5:
                print(f"   ... and {len(list(path.rglob('*'))) - 5} more items")
        else:
            print(f"❌ {dataset_name}: Missing - {path}")

def setup_datasets():
    """Main function to set up all datasets"""
    print("🚀 Setting up datasets for LSTM highlight detection...")
    
    # Create directories
    create_dataset_directories()
    
    # Download datasets
    kaggle_success = download_kaggle_dataset()
    wrestling_success = download_freestyle_wrestling_dataset()
    
    # Verify structure
    verify_dataset_structure()
    
    # Summary
    print("\n📊 Dataset Setup Summary:")
    print(f"   Kaggle Person Segmentation: {'✅' if kaggle_success else '❌'}")
    print(f"   Freestyle Wrestling: {'✅' if wrestling_success else '❌'}")
    
    if not kaggle_success:
        print("\n📋 Manual steps required:")
        print("   1. Install kaggle CLI: pip install kaggle")
        print("   2. Authenticate: kaggle login")
        print("   3. Download dataset manually from Kaggle")
    
    if not wrestling_success:
        print("\n📋 Manual steps required:")
        print("   1. Find the actual GitHub repository URL")
        print("   2. Download/clone the Freestyle-Wrestling-Dataset")
        print("   3. Place it in the datasets/freestyle_wrestling directory")

if __name__ == "__main__":
    setup_datasets() 