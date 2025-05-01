"""
Utility functions for the plant identification app.
"""

import os
import re
import json
from pathlib import Path

def format_prediction(predictions):
    """
    Format prediction results as HTML.
    
    Args:
        predictions (list): List of prediction dictionaries
        
    Returns:
        str: HTML-formatted prediction results
    """
    html = "<div class='prediction-results'>"
    
    # Add top prediction with larger styling
    top_pred = predictions[0]
    scientific_name = top_pred['class_name']
    common_name = extract_common_name(scientific_name)
    
    # If common name is in parentheses in the scientific name, extract it
    if common_name:
        name_display = f"{scientific_name.split('(')[0].strip()} <span class='common-name'>({common_name})</span>"
    else:
        name_display = scientific_name
    
    html += f"<div class='top-prediction'>"
    html += f"<h3>Identified as:</h3>"
    html += f"<p class='plant-name'>{name_display}</p>"
    html += f"<p class='confidence'>Confidence: {top_pred['probability']:.1%}</p>"
    html += "</div>"
    
    # Add other predictions if there are any
    if len(predictions) > 1:
        html += "<div class='other-predictions'>"
        html += "<h4>Other possibilities:</h4>"
        html += "<ul>"
        
        for pred in predictions[1:]:
            sci_name = pred['class_name']
            common = extract_common_name(sci_name)
            
            if common:
                display = f"{sci_name.split('(')[0].strip()} <span class='common-name'>({common})</span>"
            else:
                display = sci_name
                
            html += f"<li>{display} <span class='confidence'>({pred['probability']:.1%})</span></li>"
        
        html += "</ul>"
        html += "</div>"
    
    html += "</div>"
    return html

def extract_common_name(full_name):
    """
    Extract common name from a string with format "Scientific Name (Common Name)".
    
    Args:
        full_name (str): Full name string
        
    Returns:
        str: Common name or None if not found
    """
    match = re.search(r'\((.*?)\)', full_name)
    if match:
        return match.group(1)
    return None

def format_care_info(care_info):
    """
    Format care information as HTML.
    
    Args:
        care_info (dict): Care information dictionary
        
    Returns:
        str: HTML-formatted care information
    """
    if not care_info:
        return "<p>No care information available for this plant.</p>"
    
    common_name = care_info.get('common_name', 'Unknown')
    
    html = "<div class='care-info'>"
    
    # Add heading with common name
    html += f"<h3>{common_name} Care Guide</h3>"
    
    # Format care sections
    care_sections = [
        ('light', 'Light Requirements', '☀️'),
        ('water', 'Watering Needs', '💧'),
        ('soil', 'Soil Type', '🌱'),
        ('temperature', 'Temperature', '🌡️'),
        ('humidity', 'Humidity', '💨'),
        ('fertilizer', 'Fertilizer', '🧪'),
        ('common_issues', 'Common Issues', '⚠️')
    ]
    
    html += "<div class='care-sections'>"
    for key, label, icon in care_sections:
        if key in care_info and care_info[key]:
            html += f"<div class='care-section'>"
            html += f"<h4>{icon} {label}</h4>"
            html += f"<p>{care_info[key]}</p>"
            html += f"</div>"
    
    html += "</div>"  # close care-sections
    
    # Add toxicity warning if applicable
    if 'toxicity' in care_info and care_info['toxicity']:
        html += f"<div class='toxicity-warning'>"
        html += f"<h4>⚠️ Toxicity</h4>"
        html += f"<p>{care_info['toxicity']}</p>"
        html += f"</div>"
    
    html += "</div>"  # close care-info
    return html

def load_class_names(model_dir):
    """
    Load class names from a model directory.
    
    Args:
        model_dir (str): Directory containing the model files
        
    Returns:
        list: List of class names or None if not found
    """
    model_dir = Path(model_dir)
    
    # Try to load from class_names.json
    json_path = model_dir / 'class_names.json'
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                class_map = json.load(f)
            # Convert keys to integers and sort by key
            class_map = {int(k): v for k, v in class_map.items()}
            return [class_map[i] for i in sorted(class_map.keys())]
        except Exception as e:
            print(f"Error loading class names from JSON: {e}")
    
    # Try to load from model metadata
    model_path = model_dir / 'model.pth'
    if model_path.exists():
        try:
            import torch
            saved_data = torch.load(model_path, map_location=torch.device('cpu'))
            if 'class_names' in saved_data:
                return saved_data['class_names']
        except Exception as e:
            print(f"Error loading class names from model: {e}")
    
    return None

def get_sample_images(example_dir='examples', max_samples=6):
    """
    Get sample image paths from the examples directory.
    
    Args:
        example_dir (str): Directory containing example images
        max_samples (int): Maximum number of samples to return
        
    Returns:
        list: List of paths to sample images
    """
    example_dir = Path(example_dir)
    if not example_dir.exists():
        return []
    
    # Collect all image files with common extensions
    image_files = []
    for ext in ('.jpg', '.jpeg', '.png'):
        image_files.extend(list(example_dir.glob(f'*{ext}')))
    
    # Limit to max_samples
    return [str(path) for path in image_files[:max_samples]]

def check_dependencies():
    """
    Check if all required dependencies are installed.
    
    Returns:
        tuple: (bool success, str message)
    """
    required_packages = [
        'torch', 'torchvision', 'pillow', 'numpy', 'gradio',
        'transformers', 'tqdm', 'matplotlib'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        return False, f"Missing dependencies: {', '.join(missing)}"
    else:
        return True, "All dependencies are installed"

if __name__ == "__main__":
    # Test the utility functions
    
    # Test dependency check
    success, message = check_dependencies()
    print(message)
    
    # Test formatting functions
    test_predictions = [
        {'rank': 1, 'class_name': 'Monstera deliciosa (Swiss Cheese Plant)', 'probability': 0.92},
        {'rank': 2, 'class_name': 'Philodendron bipinnatifidum', 'probability': 0.05},
        {'rank': 3, 'class_name': 'Epipremnum aureum (Pothos)', 'probability': 0.02}
    ]
    
    test_care_info = {
        'common_name': 'Swiss Cheese Plant',
        'light': 'Bright indirect light. Avoid direct sunlight.',
        'water': 'Allow top 2-3 inches of soil to dry out between waterings.',
        'soil': 'Well-draining potting mix with peat and perlite.',
        'temperature': '65-85°F (18-29°C).',
        'humidity': 'Prefers moderate to high humidity.',
        'toxicity': 'Toxic to pets if ingested.'
    }
    
    print("\nTest prediction formatting:")
    print(format_prediction(test_predictions))
    
    print("\nTest care info formatting:")
    print(format_care_info(test_care_info))
    
    print("\nSample images:")
    print(get_sample_images())
