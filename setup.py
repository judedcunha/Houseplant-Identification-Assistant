#!/usr/bin/env python3
"""
Setup script for the Common Houseplant Identification Assistant.
This script helps set up the project, download a pre-trained model,
and prepare the environment for running the application.
"""

import os
import sys
import argparse
import subprocess
import zipfile
import shutil
import json
from pathlib import Path
import urllib.request
from tqdm import tqdm

# URL for the pre-trained model
PRETRAINED_MODEL_URL = "https://example.com/houseplant-model.zip"  # Placeholder URL

class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, output_path):
    """
    Download a file with progress bar.
    
    Args:
        url (str): URL to download
        output_path (str): Path to save the file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """
    Extract a zip file.
    
    Args:
        zip_path (str): Path to zip file
        extract_to (str): Directory to extract to
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get total size for progress bar
            total_size = sum(file.file_size for file in zip_ref.infolist())
            extracted_size = 0
            
            # Create progress bar
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Extracting") as pbar:
                for file in zip_ref.infolist():
                    zip_ref.extract(file, extract_to)
                    extracted_size += file.file_size
                    pbar.update(file.file_size)
        
        return True
    except Exception as e:
        print(f"Error extracting zip file: {e}")
        return False

def install_requirements():
    """
    Install required packages from requirements.txt.
    
    Returns:
        bool: True if successful, False otherwise
    """
    req_file = Path("requirements.txt")
    if not req_file.exists():
        print("Error: requirements.txt not found")
        return False
    
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False

def download_pretrained_model(output_dir):
    """
    Download a pre-trained model.
    
    Args:
        output_dir (str): Directory to save the model
        
    Returns:
        bool: True if successful, False otherwise
    """
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, "pretrained_model.zip")
    
    print(f"Downloading pre-trained model to {output_dir}...")
    if not download_file(PRETRAINED_MODEL_URL, zip_path):
        return False
    
    print("Extracting model files...")
    if not extract_zip(zip_path, output_dir):
        return False
    
    # Remove the zip file
    os.remove(zip_path)
    
    # Verify model files
    expected_files = ["model.pth", "class_names.json", "model_metadata.json"]
    missing_files = [f for f in expected_files if not os.path.exists(os.path.join(output_dir, f))]
    
    if missing_files:
        print(f"Warning: Missing model files: {', '.join(missing_files)}")
        return False
    
    print("Pre-trained model downloaded and extracted successfully!")
    return True

def create_model_symlink(model_path, latest_symlink):
    """
    Create a symlink to the latest model.
    
    Args:
        model_path (str): Path to the model directory
        latest_symlink (str): Path to the 'latest' symlink
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Remove existing symlink if it exists
        if os.path.exists(latest_symlink):
            if os.path.islink(latest_symlink):
                os.unlink(latest_symlink)
            else:
                shutil.rmtree(latest_symlink)
        
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(latest_symlink), exist_ok=True)
        
        # Create symlink on Unix or directory copy on Windows
        if os.name == 'posix':
            # Use relative path for the symlink
            rel_path = os.path.relpath(model_path, os.path.dirname(latest_symlink))
            os.symlink(rel_path, latest_symlink)
        else:
            # On Windows, copy the directory instead
            shutil.copytree(model_path, latest_symlink)
        
        return True
    except Exception as e:
        print(f"Error creating model symlink: {e}")
        return False

def create_examples_directory():
    """
    Create an examples directory with placeholder files.
    
    Returns:
        bool: True if successful, False otherwise
    """
    examples_dir = Path("examples")
    os.makedirs(examples_dir, exist_ok=True)
    
    # Create a README file in the examples directory
    readme_path = examples_dir / "README.txt"
    with open(readme_path, 'w') as f:
        f.write("""
Examples Directory
=================

Place sample plant images in this directory to use them as examples in the application.
The images will appear as clickable examples in the interface.

Recommended format:
- JPG or PNG format
- Clear, well-lit photos of houseplants
- File naming: plant_name.jpg (e.g., monstera_deliciosa.jpg)
""")
    
    print(f"Created examples directory at {examples_dir}")
    return True

def setup_project(download_model=True, install_deps=True):
    """
    Set up the project.
    
    Args:
        download_model (bool): Whether to download a pre-trained model
        install_deps (bool): Whether to install dependencies
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("Setting up Common Houseplant Identification Assistant...")
    
    # Create required directories
    for directory in ["data/processed", "models", "outputs"]:
        os.makedirs(directory, exist_ok=True)
    
    # Install dependencies
    if install_deps:
        if not install_requirements():
            print("Failed to install requirements")
            return False
    
    # Download pre-trained model
    if download_model:
        model_dir = "models/pretrained"
        if not download_pretrained_model(model_dir):
            print("Failed to download pre-trained model")
            return False
        
        # Create symlink to the latest model
        if not create_model_symlink(model_dir, "models/latest"):
            print("Failed to create model symlink")
            return False
    
    # Create examples directory
    create_examples_directory()
    
    print("\nSetup completed successfully!")
    print("\nTo run the application:")
    print("  python run_app.py")
    
    return True

def main():
    """Parse arguments and run the setup."""
    parser = argparse.ArgumentParser(description='Set up Houseplant Identification Assistant')
    parser.add_argument('--skip-model', action='store_true',
                        help='Skip downloading the pre-trained model')
    parser.add_argument('--skip-deps', action='store_true',
                        help='Skip installing dependencies')
    
    args = parser.parse_args()
    
    # Print welcome message
    print("=" * 60)
    print("  Common Houseplant Identification Assistant Setup")
    print("=" * 60)
    
    # Run setup
    success = setup_project(
        download_model=not args.skip_model,
        install_deps=not args.skip_deps
    )
    
    if not success:
        print("\nSetup encountered errors. Please check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
