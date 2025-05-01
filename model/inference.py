"""
Module for inference functions to identify plants from images.
"""

import os
import json
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import sys

# Import from our modules
sys.path.append('..')
from model.model_selection import get_model, get_recommended_model
from model.model_utils import load_model, predict_plant_species, load_class_names

class PlantIdentifier:
    """Class to handle plant identification"""
    
    def __init__(self, model_dir, device=None):
        """
        Initialize the plant identifier.
        
        Args:
            model_dir (str): Directory containing the model and metadata
            device: Device to run inference on (default: auto-detect)
        """
        self.model_dir = Path(model_dir)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load metadata
        self.metadata = self._load_metadata()
        self.class_names = self._load_class_names()
        self.model = self._load_model()
        
        # Check if this is a ViT model
        self.is_vit = self.metadata.get('model_type', '').lower() == 'vit'
        
        print(f"Initialized plant identifier with {len(self.class_names)} classes")
        print(f"Running on device: {self.device}")
    
    def _load_metadata(self):
        """Load model metadata"""
        meta_path = self.model_dir / 'model_metadata.json'
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                return json.load(f)
        else:
            # Try to infer basic metadata
            return {'input_size': 224}
    
    def _load_class_names(self):
        """Load class names"""
        try:
            # Try to load from JSON
            return load_class_names(self.model_dir)
        except FileNotFoundError:
            # Fall back to load from model
            model_path = self.model_dir / 'model.pth'
            if model_path.exists():
                _, metadata = load_model(model_path)
                if 'class_names' in metadata:
                    return metadata['class_names']
            
            # If all else fails, return numbered classes
            print("Warning: Could not load class names")
            return [f"Class_{i}" for i in range(self.metadata.get('num_classes', 10))]
    
    def _load_model(self):
        """Load the model"""
        # Try to load scripted model first (faster inference)
        scripted_path = self.model_dir / 'model_scripted.pt'
        if scripted_path.exists():
            try:
                model = torch.jit.load(scripted_path, map_location=self.device)
                print("Loaded TorchScript model")
                return model
            except Exception as e:
                print(f"Failed to load TorchScript model: {e}")
        
        # Fall back to regular PyTorch model
        model_path = self.model_dir / 'model.pth'
        if model_path.exists():
            # Get model type from metadata
            model_type = self.metadata.get('model_type', 'mobilenet')
            model_variant = self.metadata.get('model_variant')
            num_classes = len(self.class_names)
            
            # Create model instance
            if model_type == 'vit':
                from transformers import ViTForImageClassification
                model = ViTForImageClassification.from_pretrained(
                    'google/vit-base-patch16-224',
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True
                )
            else:
                # Use our model selection module
                model_kwargs = {}
                if model_variant:
                    if model_type == 'resnet':
                        model_kwargs['model_size'] = int(model_variant)
                    else:
                        model_kwargs['model_variant'] = model_variant
                
                model = get_model(model_type, num_classes, **model_kwargs)
            
            # Load weights
            model, _ = load_model(model_path, model)
            model = model.to(self.device)
            print(f"Loaded PyTorch {model_type} model")
            return model
        
        raise FileNotFoundError("No model found in the specified directory")
    
    def identify(self, image, top_k=3):
        """
        Identify plant in the given image.
        
        Args:
            image: PIL.Image or path to image file
            top_k (int): Number of top predictions to return
            
        Returns:
            dict: Identification results and metadata
        """
        # Load image if path is provided
        if isinstance(image, str):
            try:
                image = Image.open(image).convert('RGB')
            except Exception as e:
                raise ValueError(f"Could not open image file: {e}")
        
        # Get predictions
        predictions = predict_plant_species(
            self.model, 
            image, 
            self.class_names, 
            device=self.device, 
            top_k=top_k,
            is_vit=self.is_vit
        )
        
        # Return predictions with metadata
        return {
            'predictions': predictions,
            'top_prediction': predictions[0]['class_name'],
            'confidence': predictions[0]['probability'],
            'model_type': self.metadata.get('model_type', 'unknown'),
            'identified_at': torch.datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def get_plant_care_info(self, plant_name):
        """
        Get care information for the identified plant.
        
        Args:
            plant_name (str): Name of the plant
            
        Returns:
            dict: Care information or None if not found
        """
        # Try to load care database
        care_db_path = Path('../data/care_database.json')
        if not care_db_path.exists():
            care_db_path = Path('data/care_database.json')
        
        if care_db_path.exists():
            try:
                with open(care_db_path, 'r') as f:
                    care_db = json.load(f)
                
                # Look for exact match first
                if plant_name in care_db:
                    return care_db[plant_name]
                
                # Try common name match
                for scientific_name, info in care_db.items():
                    if info.get('common_name') == plant_name:
                        return info
                
                # Try case-insensitive partial match
                plant_name_lower = plant_name.lower()
                for scientific_name, info in care_db.items():
                    if (plant_name_lower in scientific_name.lower() or 
                        plant_name_lower in info.get('common_name', '').lower()):
                        return info
                
                # No match found
                return None
                
            except Exception as e:
                print(f"Error loading care database: {e}")
                return None
        else:
            print("Care database not found")
            return None

def identify_plant(image_path, model_dir='models/latest', top_k=3):
    """
    Standalone function to identify a plant from an image file.
    
    Args:
        image_path (str): Path to image file
        model_dir (str): Directory containing the model
        top_k (int): Number of top predictions to return
        
    Returns:
        dict: Identification results and care information
    """
    try:
        # Initialize identifier
        identifier = PlantIdentifier(model_dir)
        
        # Identify plant
        results = identifier.identify(image_path, top_k=top_k)
        
        # Get care information for top prediction
        plant_name = results['top_prediction']
        care_info = identifier.get_plant_care_info(plant_name)
        
        # Add care info to results
        results['care_info'] = care_info
        
        return results
    
    except Exception as e:
        print(f"Error identifying plant: {e}")
        return {
            'error': str(e),
            'predictions': [],
            'top_prediction': None,
            'confidence': 0.0,
            'care_info': None
        }

if __name__ == "__main__":
    """Test the plant identifier with a sample image if provided"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Identify a plant from an image')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--model-dir', type=str, default='models/latest',
                        help='Directory containing the model')
    args = parser.parse_args()
    
    if args.image:
        results = identify_plant(args.image, args.model_dir)
        
        print("\n=== Plant Identification Results ===")
        print(f"Top prediction: {results['top_prediction']}")
        print(f"Confidence: {results['confidence']:.2%}")
        
        print("\nAll predictions:")
        for pred in results['predictions']:
            print(f"  {pred['rank']}. {pred['class_name']}: {pred['probability']:.2%}")
        
        if results['care_info']:
            print("\n=== Care Information ===")
            care = results['care_info']
            print(f"Common name: {care.get('common_name', 'Unknown')}")
            print(f"Light: {care.get('light', 'Unknown')}")
            print(f"Water: {care.get('water', 'Unknown')}")
            print(f"Temperature: {care.get('temperature', 'Unknown')}")
            print(f"Humidity: {care.get('humidity', 'Unknown')}")
        else:
            print("\nNo care information available for this plant.")
    else:
        print("Please provide an image path with --image")
