import os
import json
import shutil
import glob
from pathlib import Path

""" target_species = [
    'Daucus carota', 
    'Alliaria petiolata', 
    'Hypericum perforatum', 
    'Centranthus ruber', 
    'Cirsium vulgare', 
    'Trifolium pratense', 
    'Calendula officinalis', 
    'Lamium purpureum', 
    'Alcea rosea', 
    'Papaver rhoeas'
]


target_species = [
    'Vanilla planifolia', 
    'Pelargonium graveolens', 
    'Calendula arvensis', 
    'Cirsium arvense', 
    'Lupinus polyphyllus', 
    'Tagetes lucida', 
    'Cedrela odorata', 
    'Prosopis juliflora', 
    'Acacia farnesiana', 
    'Zamia furfuracea'
]

"""

# Define the target plant species
target_species = [
    'Daucus carota', 
    'Alliaria petiolata', 
    'Hypericum perforatum', 
    'Centranthus ruber', 
    'Cirsium vulgare', 
    'Trifolium pratense', 
    'Calendula officinalis', 
    'Lamium purpureum', 
    'Alcea rosea', 
    'Papaver rhoeas'
]

# Path configurations
base_dir = '.'  # Current directory
dataset_dir = os.path.join(base_dir, 'plantnet_300K')  # Dataset directory
metadata_json = os.path.join(dataset_dir, 'plantnet300K_species_id_2_name.json')
images_dir = os.path.join(dataset_dir, 'images')
output_base_dir = os.path.join(base_dir, 'extracted_plants')  # Output directory

def load_json_metadata():
    """Load the JSON metadata file and map species names to class IDs"""
    try:
        print(f"Reading metadata from {metadata_json}...")
        with open(metadata_json, 'r', encoding='utf-8') as f:
            # The JSON file maps class IDs to species names
            id_to_species = json.load(f)
            
        # Create reverse mapping (species names to class IDs)
        species_to_ids = {}
        for class_id, species_info in id_to_species.items():
            species_name = species_info
            
            # Handle different JSON structures
            if isinstance(species_info, dict) and 'scientific_name' in species_info:
                species_name = species_info['scientific_name']
            elif isinstance(species_info, list) and len(species_info) > 0:
                species_name = species_info[0]
                
            # Add to the reverse mapping
            if species_name not in species_to_ids:
                species_to_ids[species_name] = []
            species_to_ids[species_name].append(class_id)
        
        # Filter to only include our target species
        target_species_map = {}
        for species in target_species:
            # Try exact match
            if species in species_to_ids:
                target_species_map[species] = species_to_ids[species]
            else:
                # Try partial match (e.g., if JSON has full species name with author)
                for full_species, ids in species_to_ids.items():
                    if species in full_species:
                        target_species_map[species] = ids
                        break
        
        # Print the results
        for species, class_ids in target_species_map.items():
            print(f"{species}: {len(class_ids)} class IDs found: {', '.join(class_ids)}")
            
        return target_species_map
    
    except Exception as e:
        print(f"Error reading JSON metadata: {e}")
        # Fall back to trying a directory search if the metadata file doesn't work
        return None

def find_class_dirs_by_species():
    """Search for directories that might contain the species we're looking for"""
    # First, try to use the metadata file
    species_to_ids = load_json_metadata()
    
    # If metadata loading failed, try using directory scan
    if species_to_ids is None:
        print("\nMetadata loading failed. Trying direct directory search...")
        return find_dirs_by_direct_search()
    
    return species_to_ids

def find_dirs_by_direct_search():
    """Find directories that might correspond to our target species by searching directory names"""
    species_to_dirs = {}
    
    # Check if images directory exists
    if not os.path.isdir(images_dir):
        print(f"Error: Could not find images directory at {images_dir}")
        return None
    
    print(f"\nSearching for image directories in {images_dir}...")
    
    # Look for subdirectories in the images directory
    splits = ['train', 'test', 'val']
    for split in splits:
        split_dir = os.path.join(images_dir, split)
        if not os.path.isdir(split_dir):
            continue
            
        print(f"Checking split: {split}")
        
        # List class directories in this split
        class_dirs = os.listdir(split_dir)
        for class_dir in class_dirs:
            class_path = os.path.join(split_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
                
            # Check if this class directory has any images
            image_files = glob.glob(os.path.join(class_path, '*.jpg'))
            if not image_files:
                continue
                
            # For now, just collect all directories with images
            # We'll try to match them to species later
            if 'all_dirs' not in species_to_dirs:
                species_to_dirs['all_dirs'] = []
            
            species_to_dirs['all_dirs'].append({
                'class_id': class_dir,
                'path': class_path,
                'image_count': len(image_files),
                'split': split
            })
    
    # Print summary of found directories
    if 'all_dirs' in species_to_dirs:
        found_dirs = species_to_dirs['all_dirs']
        print(f"\nFound {len(found_dirs)} class directories with images.")
        print("First few directories:")
        for i, dir_info in enumerate(found_dirs[:5]):
            print(f"  {dir_info['class_id']} ({dir_info['split']}): {dir_info['path']} - {dir_info['image_count']} images")
        
        if len(found_dirs) > 5:
            print(f"  ... and {len(found_dirs) - 5} more directories")
    else:
        print("No directories with images found.")
    
    return species_to_dirs

def extract_plant_images(species_to_ids):
    """Extract images for the target species based on the class IDs"""
    # Create output base directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process each species
    for species, class_ids in species_to_ids.items():
        # Clean species name for directory (remove any illegal chars)
        species_dir_name = species.replace(' ', '_')
        species_dir = os.path.join(output_base_dir, species_dir_name)
        
        # Create species directory
        os.makedirs(species_dir, exist_ok=True)
        
        print(f"\nProcessing {species}...")
        
        # Track if we found any images for this species
        images_found = 0
        
        # Process each class ID for this species
        for class_id in class_ids:
            images_in_class = 0
            
            # Look in all possible splits
            for split in ['train', 'test', 'val']:
                # Construct the class directory path
                class_dir = os.path.join(images_dir, split, class_id)
                
                # Check if class directory exists
                if not os.path.isdir(class_dir):
                    continue
                
                # Find all image files in this class directory
                image_files = glob.glob(os.path.join(class_dir, '*.jpg'))
                
                if image_files:
                    print(f"  Found {len(image_files)} images in {split}/{class_id}")
                    images_in_class += len(image_files)
                    
                    # Copy each image
                    for i, source_path in enumerate(image_files):
                        # Get the filename
                        filename = os.path.basename(source_path)
                        
                        # Create target path
                        target_path = os.path.join(species_dir, filename)
                        
                        try:
                            # Copy the file
                            shutil.copy2(source_path, target_path)
                            if (i+1) % 50 == 0 or i+1 == len(image_files):
                                print(f"    Copied {i+1}/{len(image_files)} images from {split}/{class_id}")
                        except Exception as e:
                            print(f"  Error copying {filename}: {e}")
            
            # Update total images found for this species
            images_found += images_in_class
            
            if images_in_class == 0:
                print(f"  No images found for class ID {class_id}")
        
        if images_found == 0:
            print(f"  No images found for {species}")
        else:
            print(f"  Total: {images_found} images copied for {species}")
    
    print("\nExtraction complete!")
    print(f"Images extracted to: {os.path.abspath(output_base_dir)}")

def create_summary_file(species_to_ids):
    """Create a summary file listing all species and their class IDs"""
    # Write summary to file
    summary_path = os.path.join(output_base_dir, 'folders_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Plant Species Class IDs and Folders\n")
        f.write("==============================\n\n")
        
        for species, class_ids in species_to_ids.items():
            f.write(f"{species}:\n")
            found_dirs = []
            
            for class_id in class_ids:
                # Check in all splits (train, test, val)
                for split in ['train', 'test', 'val']:
                    class_dir = os.path.join(images_dir, split, class_id)
                    if os.path.isdir(class_dir):
                        image_count = len(glob.glob(os.path.join(class_dir, '*.jpg')))
                        if image_count > 0:
                            found_dirs.append((class_id, split, image_count))
                
                if not any(cid == class_id for cid, _, _ in found_dirs):
                    f.write(f"  - Class ID: {class_id} (Directory not found in any split)\n")
            
            # Write found directories
            for class_id, split, image_count in found_dirs:
                f.write(f"  - Class ID: {class_id} in '{split}' split ({image_count} images)\n")
                f.write(f"    Path: {os.path.join(images_dir, split, class_id)}\n")
            
            f.write("\n")
    
    print(f"Summary file created at: {os.path.abspath(summary_path)}")

def main():
    print("Starting plant image extraction...")
    print(f"Working directory: {os.path.abspath(os.curdir)}")
    
    # Check if dataset directory exists
    if not os.path.isdir(dataset_dir):
        print(f"Error: Dataset directory not found at {dataset_dir}")
        print("Please ensure you're in the correct directory.")
        return
    
    # Get species to class IDs mapping
    species_to_ids = find_class_dirs_by_species()
    
    if not species_to_ids:
        print("Could not map species to class IDs. Aborting.")
        return
    
    # Extract images
    extract_plant_images(species_to_ids)
    
    # Create summary file
    create_summary_file(species_to_ids)
    
    print("\nDone!")

if __name__ == "__main__":
    main()