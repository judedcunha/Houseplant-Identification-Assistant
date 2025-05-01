#!/usr/bin/env python3
"""
Launcher script for the Common Houseplant Identification Assistant.
This script simplifies launching the application with various options.
"""

import os
import sys
import argparse
import subprocess
import webbrowser
import time
import platform
from pathlib import Path

def check_environment():
    """
    Check if the environment is properly set up.
    
    Returns:
        tuple: (bool success, str message)
    """
    # Check for Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        return False, f"Python 3.8+ required, but {python_version.major}.{python_version.minor} found"
    
    # Check for required directories
    required_dirs = ['app', 'data', 'model']
    for directory in required_dirs:
        if not os.path.isdir(directory):
            return False, f"Required directory '{directory}' not found"
    
    # Check for model directory
    model_dir = Path('models/latest')
    if not model_dir.exists():
        return False, "Model directory not found. Place model files in 'models/latest' or specify with --model-dir"
    
    # Try importing required packages
    try:
        import torch
        import gradio
        import PIL
    except ImportError as e:
        return False, f"Required package not found: {e}"
    
    return True, "Environment check passed"

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

def launch_app(model_dir=None, share=False, debug=False):
    """
    Launch the Gradio application.
    
    Args:
        model_dir (str, optional): Path to model directory
        share (bool): Whether to create a public link
        debug (bool): Whether to run in debug mode
        
    Returns:
        int: Process return code
    """
    # Import the app module
    try:
        sys.path.append(os.path.abspath('.'))
        from app.app import main as app_main
        
        # Run the app directly (inline)
        print("Launching the application...")
        app_main()
        return 0
        
    except ImportError as e:
        print(f"Error importing app module: {e}")
        
        # Fallback to subprocess if direct import fails
        print("Trying to launch the app using subprocess...")
        
        cmd = [sys.executable, "app/app.py"]
        if model_dir:
            cmd.extend(["--model-dir", model_dir])
        if share:
            cmd.append("--share")
            
        try:
            return subprocess.call(cmd)
        except Exception as e:
            print(f"Error launching app: {e}")
            return 1

def main():
    """Main function to parse arguments and launch the application."""
    parser = argparse.ArgumentParser(description='Launch Houseplant Identification Assistant')
    parser.add_argument('--model-dir', type=str, default=None,
                        help='Directory containing the model (default: models/latest)')
    parser.add_argument('--share', action='store_true',
                        help='Create a public link for the application')
    parser.add_argument('--install-deps', action='store_true',
                        help='Install dependencies from requirements.txt')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Print welcome message
    print("=" * 60)
    print("  Common Houseplant Identification Assistant")
    print("=" * 60)
    
    # Install dependencies if requested
    if args.install_deps:
        if not install_requirements():
            sys.exit(1)
    
    # Check environment
    success, message = check_environment()
    if not success:
        print(f"Environment check failed: {message}")
        print("Run with --install-deps to install required dependencies")
        sys.exit(1)
    
    print(message)
    
    # Launch the application
    print("Starting the application. Press Ctrl+C to exit.")
    return_code = launch_app(args.model_dir, args.share, args.debug)
    
    # Exit with the return code
    sys.exit(return_code)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nApplication terminated by user")
        sys.exit(0)
