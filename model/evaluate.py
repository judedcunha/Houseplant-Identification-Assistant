"""
Script to evaluate a trained plant identification model on test data.
Generates detailed metrics and visualizations for model performance.
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from model.model_utils import load_model, load_class_names
from model.model_selection import get_model
from model.inference import PlantIdentifier
from data.preprocess_images import preprocess_single_image

def load_test_data(test_dir):
    """
    Load test images and their true labels.
    
    Args:
        test_dir (str): Directory containing test images organized in class folders
        
    Returns:
        tuple: (image_paths, true_labels, class_names)
    """
    test_dir = Path(test_dir)
    
    # Get all class directories
    class_dirs = [d for d in test_dir.iterdir() if d.is_dir()]
    class_names = [d.name for d in class_dirs]
    
    # Map class names to indices
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    # Collect image paths and labels
    image_paths = []
    true_labels = []
    
    for class_dir in class_dirs:
        class_idx = class_to_idx[class_dir.name]
        
        # Get all image files
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files = list(class_dir.glob(f'*{ext}')) + list(class_dir.glob(f'*{ext.upper()}'))
            
            for img_path in image_files:
                image_paths.append(str(img_path))
                true_labels.append(class_idx)
    
    return image_paths, true_labels, class_names

def evaluate_model(model_dir, test_dir, output_dir=None, batch_size=32):
    """
    Evaluate model performance on test data.
    
    Args:
        model_dir (str): Directory containing the trained model
        test_dir (str): Directory containing test images
        output_dir (str, optional): Directory to save evaluation results
        batch_size (int): Batch size for evaluation
        
    Returns:
        dict: Evaluation metrics
    """
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize plant identifier
    identifier = PlantIdentifier(model_dir)
    model = identifier.model
    class_names = identifier.class_names
    device = identifier.device
    
    print(f"Loaded model with {len(class_names)} classes")
    print(f"Using device: {device}")
    
    # Load test data
    print(f"Loading test data from {test_dir}")
    image_paths, true_labels, test_class_names = load_test_data(test_dir)
    
    # If test_class_names and class_names differ, align them
    if set(test_class_names) != set(class_names):
        print("Warning: Test class names differ from model class names")
        print(f"Model classes: {len(class_names)}")
        print(f"Test classes: {len(test_class_names)}")
        
        # Create a mapping from test class names to model class indices
        test_to_model = {}
        for test_idx, test_name in enumerate(test_class_names):
            if test_name in class_names:
                model_idx = class_names.index(test_name)
                test_to_model[test_idx] = model_idx
            else:
                print(f"Warning: Test class '{test_name}' not found in model classes")
        
        # Remap true labels
        remapped_labels = []
        valid_indices = []
        for i, label in enumerate(true_labels):
            if label in test_to_model:
                remapped_labels.append(test_to_model[label])
                valid_indices.append(i)
            else:
                print(f"Skipping image with label {label}")
        
        true_labels = remapped_labels
        image_paths = [image_paths[i] for i in valid_indices]
    
    num_samples = len(image_paths)
    print(f"Evaluating on {num_samples} test images")
    
    # Set model to evaluation mode
    model.eval()
    
    # Run inference in batches
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Evaluating"):
            batch_paths = image_paths[i:i + batch_size]
            batch_results = []
            
            # Process each image in the batch
            for img_path in batch_paths:
                result = identifier.identify(img_path)
                batch_results.append(result)
            
            # Extract predictions and probabilities
            for result in batch_results:
                # Get the predicted class index
                pred_class = result['top_prediction']
                pred_idx = class_names.index(pred_class)
                all_predictions.append(pred_idx)
                
                # Get the top probability
                all_probabilities.append(result['confidence'])
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, all_predictions)
    report = classification_report(
        true_labels, all_predictions, 
        target_names=class_names, 
        output_dict=True
    )
    
    print(f"Overall accuracy: {accuracy:.2%}")
    
    # Save results if output directory is specified
    if output_dir:
        # Save classification report
        report_path = os.path.join(output_dir, 'classification_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        # Plot and save confusion matrix
        cm = confusion_matrix(true_labels, all_predictions)
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        
        # Plot and save per-class accuracy
        per_class_acc = [report[name]['precision'] for name in class_names]
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(per_class_acc)), per_class_acc)
        plt.xticks(range(len(per_class_acc)), class_names, rotation=90)
        plt.ylabel('Accuracy')
        plt.title('Per-Class Accuracy')
        
        # Add the values on top of the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom'
            )
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'per_class_accuracy.png'))
        
        # Plot and save confidence distribution
        plt.figure(figsize=(10, 6))
        plt.hist(all_probabilities, bins=20)
        plt.xlabel('Confidence')
        plt.ylabel('Number of Predictions')
        plt.title('Confidence Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'))
        
        # Save summary metrics
        summary = {
            'accuracy': accuracy,
            'number_of_test_samples': num_samples,
            'number_of_classes': len(class_names),
            'mean_confidence': np.mean(all_probabilities),
            'median_confidence': np.median(all_probabilities),
            'min_confidence': min(all_probabilities),
            'max_confidence': max(all_probabilities)
        }
        
        with open(os.path.join(output_dir, 'summary_metrics.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"Evaluation results saved to {output_dir}")
    
    # Return metrics
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist() if 'cm' in locals() else None,
        'predictions': all_predictions,
        'true_labels': true_labels,
        'probabilities': all_probabilities
    }

def main():
    """Parse arguments and run the evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate plant identification model')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory containing the trained model')
    parser.add_argument('--test-dir', type=str, required=True,
                        help='Directory containing test images organized in class folders')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save evaluation results')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_model(
        args.model_dir,
        args.test_dir,
        args.output_dir,
        args.batch_size
    )

if __name__ == "__main__":
    main()
