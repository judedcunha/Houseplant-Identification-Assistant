"""
Test script to verify the plant identification model and inference pipeline.
This can be used to test the model before deploying the full application.
"""

import os
import sys
import argparse
from pathlib import Path
import torch
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from model.inference import PlantIdentifier
from data.preprocess_images import preprocess_single_image

def test_identification(image_path, model_dir, top_k=3):
    """
    Test plant identification on a single image.
    
    Args:
        image_path (str): Path to the test image
        model_dir (str): Directory containing the model
        top_k (int): Number of top predictions to display
    """
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    print(f"Testing plant identification on {image_path}")
    print(f"Using model from {model_dir}")
    
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        print(f"Loaded image of size {image.size}")
        
        # Initialize plant identifier
        identifier = PlantIdentifier(model_dir)
        print(f"Loaded model with {len(identifier.class_names)} classes")
        
        # Run identification
        results = identifier.identify(image, top_k=top_k)
        
        # Display results
        print("\n=== Identification Results ===")
        print(f"Top prediction: {results['top_prediction']}")
        print(f"Confidence: {results['confidence']:.2%}")
        
        print("\nAll predictions:")
        for pred in results['predictions']:
            print(f"  {pred['rank']}. {pred['class_name']}: {pred['probability']:.2%}")
        
        # Get care information
        print("\n=== Care Information ===")
        plant_name = results['top_prediction']
        care_info = identifier.get_plant_care_info(plant_name)
        
        if care_info:
            common_name = care_info.get('common_name', 'Unknown')
            print(f"Common name: {common_name}")
            print(f"Light: {care_info.get('light', 'Unknown')}")
            print(f"Water: {care_info.get('water', 'Unknown')}")
            print(f"Temperature: {care_info.get('temperature', 'Unknown')}")
            print(f"Humidity: {care_info.get('humidity', 'Unknown')}")
            
            if 'toxicity' in care_info:
                print(f"\nToxicity warning: {care_info['toxicity']}")
        else:
            print("No care information available for this plant.")
        
        print("\nTest completed successfully!")
        return results
        
    except Exception as e:
        print(f"Error during identification: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to parse arguments and run the test."""
    parser = argparse.ArgumentParser(description='Test plant identification')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to image file')
    parser.add_argument('--model-dir', type=str, default='models/latest',
                        help='Directory containing the model')
    parser.add_argument('--top-k', type=int, default=3,
                        help='Number of top predictions to display')
    
    args = parser.parse_args()
    
    # Run test
    test_identification(args.image, args.model_dir, args.top_k)

if __name__ == "__main__":
    main()
