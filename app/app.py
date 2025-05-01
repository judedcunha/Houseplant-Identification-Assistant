"""
Gradio web application for the Common Houseplant Identification Assistant.
"""

import os
import sys
import json
import torch
from PIL import Image
import gradio as gr
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from our modules
from model.inference import PlantIdentifier
from app.utils import format_prediction, format_care_info
from app.styling import get_css

# Default paths
DEFAULT_MODEL_DIR = "../models/latest"
CARE_DB_PATH = "../data/care_database.json"

class PlantApp:
    """Class to handle the Gradio app for plant identification"""
    
    def __init__(self, model_dir=DEFAULT_MODEL_DIR, care_db_path=CARE_DB_PATH):
        """
        Initialize the plant identification app.
        
        Args:
            model_dir (str): Directory containing the model
            care_db_path (str): Path to care database JSON file
        """
        self.model_dir = Path(model_dir)
        self.care_db_path = Path(care_db_path)
        
        # Load care database
        self.care_db = self._load_care_database()
        
        # Initialize plant identifier
        try:
            self.identifier = PlantIdentifier(self.model_dir)
            self.model_ready = True
            print(f"Model loaded successfully with {len(self.identifier.class_names)} classes")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_ready = False
            self.identifier = None
    
    def _load_care_database(self):
        """Load the plant care database."""
        try:
            if self.care_db_path.exists():
                with open(self.care_db_path, 'r') as f:
                    return json.load(f)
            else:
                # Try alternative path
                alt_path = Path("data/care_database.json")
                if alt_path.exists():
                    with open(alt_path, 'r') as f:
                        return json.load(f)
                else:
                    print("Care database not found")
                    return {}
        except Exception as e:
            print(f"Error loading care database: {e}")
            return {}
    
    def identify_plant(self, image, top_k=3):
        """
        Identify plant in the given image.
        
        Args:
            image: PIL.Image or numpy array or path to image file
            top_k (int): Number of top predictions to return
            
        Returns:
            tuple: (formatted predictions, care information, confidence)
        """
        if not self.model_ready:
            return "Model not loaded properly. Please check the logs.", "", 0
        
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Get identification results
            results = self.identifier.identify(image, top_k=top_k)
            
            # Get the scientific name
            top_prediction = results['top_prediction']
            
            # Extract scientific name if it's in "Scientific Name (Common Name)" format
            scientific_name = top_prediction
            if "(" in scientific_name:
                scientific_name = scientific_name.split("(")[0].strip()
            
            # Get care information
            care_info = None
            
            # Look for exact match first
            if scientific_name in self.care_db:
                care_info = self.care_db[scientific_name]
            else:
                # Try case-insensitive partial match
                scientific_name_lower = scientific_name.lower()
                for name, info in self.care_db.items():
                    if (scientific_name_lower in name.lower() or 
                        scientific_name_lower in info.get('common_name', '').lower()):
                        care_info = info
                        break
            
            # Format results
            predictions_html = format_prediction(results['predictions'])
            care_html = format_care_info(care_info) if care_info else "No care information available for this plant."
            confidence = float(results['confidence']) * 100  # Convert to percentage
            
            return predictions_html, care_html, confidence
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error identifying plant: {str(e)}", "", 0
    
    def create_interface(self):
        """Create the Gradio interface."""
        # Title and description
        title = "Common Houseplant Identification Assistant"
        description = """
        Upload an image of your houseplant, and this tool will identify the species and provide care recommendations.
        The model can recognize around 20 common houseplant species.
        """
        
        # Create the interface
        with gr.Blocks(css=get_css()) as interface:
            gr.Markdown(f"<h1 class='title'>{title}</h1>")
            gr.Markdown(f"<p class='description'>{description}</p>")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Input components
                    input_image = gr.Image(
                        type="pil",
                        label="Upload or take a photo of your houseplant"
                    )
                    identify_button = gr.Button("Identify Plant", variant="primary")
                    
                with gr.Column(scale=1):
                    # Output components
                    confidence_meter = gr.Number(
                        label="Confidence (%)", 
                        value=0,
                        interactive=False
                    )
                    identification_result = gr.HTML(
                        label="Identification Results",
                        value="Upload an image and click 'Identify Plant' to get results."
                    )
                    care_info = gr.HTML(
                        label="Care Information",
                        value="Plant care information will appear here after identification."
                    )
            
            # Set up the action
            identify_button.click(
                fn=self.identify_plant,
                inputs=[input_image],
                outputs=[identification_result, care_info, confidence_meter]
            )
            
            # Example images
            example_dir = Path("examples")
            examples = []
            
            if example_dir.exists():
                example_files = list(example_dir.glob("*.jpg")) + list(example_dir.glob("*.png"))
                if example_files:
                    examples = [[str(file)] for file in example_files[:6]]  # Limit to 6 examples
            
            if examples:
                gr.Examples(
                    examples=examples,
                    inputs=input_image,
                    outputs=[identification_result, care_info, confidence_meter],
                    fn=self.identify_plant
                )
            
            # Footer info
            gr.Markdown("""
            <div class='footer'>
                <p>This tool can identify around 20 common houseplant species. It works best with clear, well-lit photos 
                showing the plant's leaves and overall structure. The identification is based on visual features only, 
                so accuracy may vary based on the plant's condition and the photo quality.</p>
                <p>For more accurate identification, consider:</p>
                <ul>
                    <li>Taking photos in natural light</li>
                    <li>Capturing multiple angles of the plant</li>
                    <li>Including both close-ups of leaves and full plant shots</li>
                </ul>
            </div>
            """)
        
        return interface

def main():
    """Main function to launch the app."""
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Launch the Houseplant Identification Assistant')
    parser.add_argument('--model-dir', type=str, default=DEFAULT_MODEL_DIR,
                        help='Directory containing the model')
    parser.add_argument('--care-db', type=str, default=CARE_DB_PATH,
                        help='Path to care database JSON file')
    parser.add_argument('--share', action='store_true',
                        help='Create a public link for the interface')
    args = parser.parse_args()
    
    # Create the app
    app = PlantApp(args.model_dir, args.care_db)
    interface = app.create_interface()
    
    # Launch the interface
    interface.launch(share=args.share)

if __name__ == "__main__":
    main()
