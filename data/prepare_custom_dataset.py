"""
Utility to help users prepare their own custom datasets for training.
This tool can process folders of plant images and organize them for training.
"""

import os
import sys
import argparse
import shutil
import json
from pathlib import Path
import random
from PIL import Image
from tqdm import tqdm

def validate_image(image_path):
    """
    Validate if a file is a valid image.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        bool: True if it's a valid image, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def create_dataset_structure(output_dir):
    """
    Create the necessary directory structure for training.
    
    Args:
        output_dir (str): Base output directory
        
    Returns:
        tuple: (train_dir, val_dir) paths
    """
    train_dir = os.path.join(output_dir, "processed", "train")
    val_dir = os.path.join(output_dir, "processed", "val")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    print(f"Created directory structure in {output_dir}")
    return train_dir, val_dir

def process_source_directory(source_dir, train_dir, val_dir, split_ratio=0.2, min_images=5):
    """
    Process a directory of plant images and split into train/validation sets.
    
    Args:
        source_dir (str): Source directory with plant images
        train_dir (str): Output directory for training images
        val_dir (str): Output directory for validation images
        split_ratio (float): Validation split ratio (0.0-1.0)
        min_images (int): Minimum number of images required per class
        
    Returns:
        tuple: (class_name, num_processed, num_skipped)
    """
    source_path = Path(source_dir)
    class_name = source_path.name
    
    # Create class directories
    train_class_dir = os.path.join(train_dir, class_name)
    val_class_dir = os.path.join(val_dir, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)
    
    # Get all image files
    image_files = []
    for extension in ['.jpg', '.jpeg', '.png']:
        image_files.extend(list(source_path.glob(f"*{extension}")))
        image_files.extend(list(source_path.glob(f"*{extension.upper()}")))
    
    # Check if we have enough images
    if len(image_files) < min_images:
        print(f"Warning: {class_name} has only {len(image_files)} images (minimum {min_images})")
        return class_name, 0, len(image_files)
    
    # Validate and collect good images
    valid_images = []
    skipped = 0
    
    for img_path in tqdm(image_files, desc=f"Validating {class_name}"):
        if validate_image(img_path):
            valid_images.append(img_path)
        else:
            skipped += 1
    
    # Shuffle and split
    random.shuffle(valid_images)
    split_idx = int(len(valid_images) * (1 - split_ratio))
    
    train_images = valid_images[:split_idx]
    val_images = valid_images[split_idx:]
    
    # Copy the images
    for idx, img_path in enumerate(tqdm(train_images, desc=f"Copying {class_name} train")):
        dest_path = os.path.join(train_class_dir, f"{class_name}_{idx:04d}{img_path.suffix}")
        shutil.copy2(img_path, dest_path)
    
    for idx, img_path in enumerate(tqdm(val_images, desc=f"Copying {class_name} val")):
        dest_path = os.path.join(val_class_dir, f"{class_name}_{idx:04d}{img_path.suffix}")
        shutil.copy2(img_path, dest_path)
    
    return class_name, len(valid_images), skipped

def create_class_mapping(output_dir, class_info):
    """
    Create a JSON file mapping class names to readable labels.
    
    Args:
        output_dir (str): Base output directory
        class_info (dict): Dictionary of class information
        
    Returns:
        str: Path to the created JSON file
    """
    mapping_path = os.path.join(output_dir, "processed", "label_mapping.json")
    
    # Create label mapping
    label_map = {}
    for idx, (class_name, info) in enumerate(class_info.items()):
        if info['processed'] > 0:
            label_map[idx] = {
                "directory": class_name,
                "scientific_name": info['scientific_name'],
                "common_name": info['common_name']
            }
    
    # Save the mapping
    with open(mapping_path, 'w') as f:
        json.dump(label_map, f, indent=4)
    
    print(f"Created label mapping with {len(label_map)} classes")
    return mapping_path

def prepare_custom_dataset(source_dir, output_dir, split_ratio=0.2, min_images=5):
    """
    Main function to prepare a custom dataset from a source directory.
    
    Args:
        source_dir (str): Source directory with subdirectories for each plant class
        output_dir (str): Output directory for the processed dataset
        split_ratio (float): Validation split ratio (0.0-1.0)
        min_images (int): Minimum number of images required per class
        
    Returns:
        tuple: (train_dir, val_dir, label_mapping_path)
    """
    print(f"Preparing custom dataset from {source_dir}")
    
    # Create directory structure
    train_dir, val_dir = create_dataset_structure(output_dir)
    
    # Get all subdirectories (each represents a class)
    class_dirs = [d for d in os.listdir(source_dir) 
                  if os.path.isdir(os.path.join(source_dir, d))]
    
    if not class_dirs:
        print(f"Error: No class directories found in {source_dir}")
        print("Each plant class should be in its own subdirectory")
        return None, None, None
    
    print(f"Found {len(class_dirs)} classes")
    
    # Process each class directory
    class_info = {}
    for class_dir in class_dirs:
        source_class_path = os.path.join(source_dir, class_dir)
        class_name, processed, skipped = process_source_directory(
            source_class_path, train_dir, val_dir, split_ratio, min_images
        )
        
        # Store information about the class
        # Try to extract scientific name and common name from directory name
        parts = class_name.split('_')
        if len(parts) >= 2:
            scientific_name = ' '.join(parts[:2])  # First two parts for genus + species
            common_name = ' '.join(parts[2:])      # Remaining parts for common name
            if not common_name:
                common_name = scientific_name      # Default to scientific name
        else:
            scientific_name = class_name
            common_name = class_name
        
        class_info[class_name] = {
            'scientific_name': scientific_name,
            'common_name': common_name,
            'processed': processed,
            'skipped': skipped
        }
    
    # Create label mapping
    label_mapping_path = create_class_mapping(output_dir, class_info)
    
    # Print summary
    print("\n=== Dataset Preparation Summary ===")
    total_processed = sum(info['processed'] for info in class_info.values())
    total_skipped = sum(info['skipped'] for info in class_info.values())
    
    print(f"Total processed images: {total_processed}")
    print(f"Total skipped images: {total_skipped}")
    print(f"Total classes: {len([c for c, info in class_info.items() if info['processed'] > 0])}")
    print(f"Train directory: {train_dir}")
    print(f"Validation directory: {val_dir}")
    print(f"Label mapping: {label_mapping_path}")
    
    return train_dir, val_dir, label_mapping_path

def main():
    """Parse arguments and run the dataset preparation."""
    parser = argparse.ArgumentParser(description='Prepare custom training dataset')
    parser.add_argument('--source', type=str, required=True,
                        help='Source directory with subdirectories for each plant class')
    parser.add_argument('--output', type=str, default='data',
                        help='Output directory for the processed dataset')
    parser.add_argument('--split', type=float, default=0.2,
                        help='Validation split ratio (0.0-1.0)')
    parser.add_argument('--min-images', type=int, default=5,
                        help='Minimum number of images required per class')
    
    args = parser.parse_args()
    
    # Check source directory
    if not os.path.isdir(args.source):
        print(f"Error: Source directory {args.source} not found")
        sys.exit(1)
    
    # Prepare the dataset
    prepare_custom_dataset(
        args.source, args.output, args.split, args.min_images
    )

if __name__ == "__main__":
    main()
