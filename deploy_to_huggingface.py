"""
Script to deploy the Houseplant Identification Assistant to Hugging Face Spaces.
This simplifies the process of creating a Hugging Face Space with the application.
"""

import os
import sys
import argparse
import shutil
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

def prepare_deployment_folder(model_dir, output_dir):
    """
    Prepare a folder with all files needed for deployment.
    
    Args:
        model_dir (str): Path to the model directory
        output_dir (str): Directory to save deployment files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Preparing deployment files in {output_dir}")
    
    # Copy necessary files and directories
    files_to_copy = [
        "app/app.py",
        "app/utils.py",
        "app/styling.py",
        "model/inference.py",
        "model/model_selection.py",
        "model/model_utils.py",
        "data/care_database.json",
        "data/preprocess_images.py",
        "requirements.txt",
        "README.md"
    ]
    
    for file in files_to_copy:
        src_path = Path(file)
        dst_path = Path(output_dir) / src_path
        
        # Create parent directories if needed
        os.makedirs(dst_path.parent, exist_ok=True)
        
        # Copy file
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            print(f"Copied {src_path} to {dst_path}")
        else:
            print(f"Warning: {src_path} not found")
    
    # Create models directory
    models_dir = Path(output_dir) / "models" / "latest"
    os.makedirs(models_dir, exist_ok=True)
    
    # Copy model files
    model_files = [
        "model.pth",
        "model_scripted.pt",
        "class_names.json",
        "model_metadata.json"
    ]
    
    for file in model_files:
        src_path = Path(model_dir) / file
        if src_path.exists():
            shutil.copy2(src_path, models_dir / file)
            print(f"Copied model file {file}")
        else:
            print(f"Warning: Model file {file} not found")
    
    # Create app.py in the root directory that imports from the app directory
    with open(os.path.join(output_dir, "app.py"), "w") as f:
        f.write("""
# Main app file for Hugging Face Spaces deployment
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the app
from app.app import main

if __name__ == "__main__":
    main()
""")
    
    # Create requirements-spaces.txt with specific versions for Hugging Face
    with open(os.path.join(output_dir, "requirements.txt"), "r") as f:
        requirements = f.readlines()
    
    # Add huggingface_hub to requirements
    requirements.append("huggingface_hub>=0.12.0\n")
    
    with open(os.path.join(output_dir, "requirements.txt"), "w") as f:
        f.writelines(requirements)
    
    # Create a minimal .gitignore
    with open(os.path.join(output_dir, ".gitignore"), "w") as f:
        f.write("""
__pycache__/
*.py[cod]
*$py.class
.env
.venv
env/
venv/
ENV/
.ipynb_checkpoints
.DS_Store
""")
    
    print("Deployment files prepared successfully!")
    return output_dir

def deploy_to_huggingface(deployment_dir, hf_token, space_name=None, space_visibility="public"):
    """
    Deploy the application to Hugging Face Spaces.
    
    Args:
        deployment_dir (str): Directory with deployment files
        hf_token (str): Hugging Face API token
        space_name (str, optional): Name for the Hugging Face Space
        space_visibility (str): Visibility of the space ('public' or 'private')
    """
    if not hf_token:
        print("Error: Hugging Face token is required for deployment")
        print("Get your token at https://huggingface.co/settings/tokens")
        return
    
    # Initialize Hugging Face API
    api = HfApi(token=hf_token)
    
    # Determine space name if not provided
    if not space_name:
        # Use the directory name of the deployment folder
        space_name = os.path.basename(os.path.abspath(deployment_dir))
        # Ensure it's valid for HF
        space_name = space_name.replace("_", "-").lower()
    
    # Add username prefix if not present
    if "/" not in space_name:
        user_info = api.whoami()
        username = user_info["name"]
        space_name = f"{username}/{space_name}"
    
    print(f"Deploying to Hugging Face Space: {space_name}")
    
    try:
        # Create the repository
        create_repo(
            repo_id=space_name,
            token=hf_token,
            repo_type="space",
            space_sdk="gradio",
            private=space_visibility == "private"
        )
        print(f"Created Hugging Face Space: {space_name}")
        
        # Upload the files
        upload_folder(
            folder_path=deployment_dir,
            repo_id=space_name,
            repo_type="space",
            token=hf_token
        )
        
        print(f"Deployed successfully to: https://huggingface.co/spaces/{space_name}")
        return f"https://huggingface.co/spaces/{space_name}"
        
    except Exception as e:
        print(f"Error during deployment: {e}")
        return None

def main():
    """Main function to parse arguments and deploy the application."""
    parser = argparse.ArgumentParser(description='Deploy to Hugging Face Spaces')
    parser.add_argument('--model-dir', type=str, default='models/latest',
                        help='Directory containing the model files')
    parser.add_argument('--output-dir', type=str, default='deployment',
                        help='Directory to prepare deployment files')
    parser.add_argument('--space-name', type=str, default=None,
                        help='Name for the Hugging Face Space (username/space-name)')
    parser.add_argument('--token', type=str, default=None,
                        help='Hugging Face API token')
    parser.add_argument('--visibility', type=str, default='public',
                        choices=['public', 'private'],
                        help='Visibility of the Hugging Face Space')
    
    args = parser.parse_args()
    
    # Check for token in environment variable if not provided
    hf_token = args.token or os.environ.get("HF_TOKEN")
    
    if not hf_token:
        print("Error: Hugging Face token is required for deployment")
        print("Either provide it with --token or set the HF_TOKEN environment variable")
        sys.exit(1)
    
    # Prepare deployment files
    deployment_dir = prepare_deployment_folder(args.model_dir, args.output_dir)
    
    # Deploy to Hugging Face
    deploy_to_huggingface(deployment_dir, hf_token, args.space_name, args.visibility)

if __name__ == "__main__":
    main()
