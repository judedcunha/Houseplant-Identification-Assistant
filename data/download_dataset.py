"""
Script to download and organize houseplant dataset from various sources.
Primarily uses a subset of the PlantNet dataset, focusing on common houseplants.
"""

import os
import json
import shutil
import requests
from tqdm import tqdm
from huggingface_hub import snapshot_download

# List of common houseplant species to include
COMMON_HOUSEPLANTS = [
    "Monstera deliciosa",        # Swiss Cheese Plant
    "Ficus lyrata",              # Fiddle Leaf Fig
    "Sansevieria trifasciata",   # Snake Plant
    "Chlorophytum comosum",      # Spider Plant
    "Epipremnum aureum",         # Pothos
    "Spathiphyllum wallisii",    # Peace Lily
    "Zamioculcas zamiifolia",    # ZZ Plant
    "Dracaena marginata",        # Dragon Tree
    "Calathea makoyana",         # Peacock Plant
    "Pilea peperomioides",       # Chinese Money Plant
    "Philodendron bipinnatifidum", # Split-leaf Philodendron
    "Aloe vera",                 # Aloe Vera
    "Ficus elastica",            # Rubber Plant
    "Maranta leuconeura",        # Prayer Plant
    "Aglaonema commutatum",      # Chinese Evergreen
    "Peperomia obtusifolia",     # Baby Rubber Plant
    "Anthurium andraeanum",      # Flamingo Flower
    "Schlumbergera bridgesii",   # Christmas Cactus
    "Crassula ovata",            # Jade Plant
    "Aspidistra elatior"         # Cast Iron Plant
]

def create_directory_structure():
    """Create the necessary directory structure for the dataset."""
    # Create main data directory
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/processed/train", exist_ok=True)
    os.makedirs("data/processed/val", exist_ok=True)
    
    # Create directories for each plant species
    for plant in COMMON_HOUSEPLANTS:
        # Replace spaces with underscores for directory names
        plant_dir = plant.replace(" ", "_").lower()
        os.makedirs(f"data/processed/train/{plant_dir}", exist_ok=True)
        os.makedirs(f"data/processed/val/{plant_dir}", exist_ok=True)

def download_plantnet_subset():
    """
    Download a subset of the PlantNet dataset from Hugging Face that contains
    images of common houseplants.
    """
    print("Downloading PlantNet subset from Hugging Face...")
    
    try:
        # Download the entire PlantNet dataset or a subset if available
        dataset_path = snapshot_download(
            repo_id="plantnet/plantnet_300K",
            repo_type="dataset",
            local_dir="./data/raw/plantnet"
        )
        print(f"Dataset downloaded to {dataset_path}")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        
        # Alternative approach if Hugging Face download fails
        print("Attempting alternative download method...")
        try:
            # This is a fallback - in a real implementation, you would need to 
            # point to an actual data source with the PlantNet subset
            url = "https://example.com/plantnet_houseplants_subset.zip"
            response = requests.get(url, stream=True)
            
            if response.status_code == 200:
                with open("./data/raw/plantnet_subset.zip", "wb") as f:
                    total_size = int(response.headers.get("content-length", 0))
                    with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                
                # Extract the zip file
                import zipfile
                with zipfile.ZipFile("./data/raw/plantnet_subset.zip", "r") as zip_ref:
                    zip_ref.extractall("./data/raw/plantnet")
                
                return True
            else:
                print(f"Failed to download. Status code: {response.status_code}")
                return False
        except Exception as e:
            print(f"Error in alternative download: {e}")
            return False

def organize_dataset():
    """
    Organize the downloaded PlantNet data into our project structure,
    selecting only the common houseplants we're interested in.
    """
    print("Organizing dataset...")
    
    # Map PlantNet species names to our standard names if needed
    plantnet_to_standard = {
        # Add mappings if PlantNet uses different scientific names
        # "plantnet_name": "our_standard_name"
    }
    
    # Default to same name if not in mapping
    for plant in COMMON_HOUSEPLANTS:
        if plant not in plantnet_to_standard.values():
            plant_key = plant.lower()
            plantnet_to_standard[plant_key] = plant
    
    # Process each plant species
    for plantnet_name, standard_name in plantnet_to_standard.items():
        standard_dir = standard_name.replace(" ", "_").lower()
        plantnet_dir = plantnet_name.replace(" ", "_").lower()
        
        # Source directory in PlantNet dataset
        source_dir = f"./data/raw/plantnet/{plantnet_dir}"
        
        if os.path.exists(source_dir):
            print(f"Processing {standard_name}...")
            
            # Get all image files
            image_files = []
            for root, _, files in os.walk(source_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_files.append(os.path.join(root, file))
            
            # Split into train (80%) and validation (20%)
            from sklearn.model_selection import train_test_split
            train_files, val_files = train_test_split(
                image_files, test_size=0.2, random_state=42
            )
            
            # Copy to our directory structure
            for src_file in tqdm(train_files, desc="Copying train images"):
                dst_file = f"data/processed/train/{standard_dir}/{os.path.basename(src_file)}"
                shutil.copy2(src_file, dst_file)
            
            for src_file in tqdm(val_files, desc="Copying validation images"):
                dst_file = f"data/processed/val/{standard_dir}/{os.path.basename(src_file)}"
                shutil.copy2(src_file, dst_file)
        else:
            print(f"Warning: No data found for {standard_name} at {source_dir}")
    
    print("Dataset organization completed.")

def create_label_mapping():
    """Create a JSON file mapping directory names to readable plant names."""
    print("Creating label mapping...")
    
    label_map = {}
    class_idx = 0
    
    for plant in COMMON_HOUSEPLANTS:
        dir_name = plant.replace(" ", "_").lower()
        common_name = get_common_name(plant)
        
        label_map[class_idx] = {
            "directory": dir_name,
            "scientific_name": plant,
            "common_name": common_name
        }
        class_idx += 1
    
    # Save the mapping
    with open("data/processed/label_mapping.json", "w") as f:
        json.dump(label_map, f, indent=4)
    
    print(f"Label mapping saved with {len(label_map)} plant species.")
    return label_map

def get_common_name(scientific_name):
    """Get the common name for a scientific plant name."""
    # Mapping of scientific names to common names
    name_map = {
        "Monstera deliciosa": "Swiss Cheese Plant",
        "Ficus lyrata": "Fiddle Leaf Fig",
        "Sansevieria trifasciata": "Snake Plant",
        "Chlorophytum comosum": "Spider Plant",
        "Epipremnum aureum": "Pothos",
        "Spathiphyllum wallisii": "Peace Lily",
        "Zamioculcas zamiifolia": "ZZ Plant",
        "Dracaena marginata": "Dragon Tree",
        "Calathea makoyana": "Peacock Plant",
        "Pilea peperomioides": "Chinese Money Plant",
        "Philodendron bipinnatifidum": "Split-leaf Philodendron",
        "Aloe vera": "Aloe Vera",
        "Ficus elastica": "Rubber Plant",
        "Maranta leuconeura": "Prayer Plant",
        "Aglaonema commutatum": "Chinese Evergreen",
        "Peperomia obtusifolia": "Baby Rubber Plant",
        "Anthurium andraeanum": "Flamingo Flower",
        "Schlumbergera bridgesii": "Christmas Cactus",
        "Crassula ovata": "Jade Plant",
        "Aspidistra elatior": "Cast Iron Plant"
    }
    
    return name_map.get(scientific_name, "Unknown")

def main():
    """Main function to execute the data download and organization process."""
    print("Starting dataset download and organization...")
    
    # Create directory structure
    create_directory_structure()
    
    # Download dataset
    if download_plantnet_subset():
        # Organize the dataset
        organize_dataset()
        
        # Create label mapping
        create_label_mapping()
        
        print("Dataset preparation completed successfully!")
    else:
        print("Dataset download failed. Please check your connection or try again later.")

if __name__ == "__main__":
    main()
