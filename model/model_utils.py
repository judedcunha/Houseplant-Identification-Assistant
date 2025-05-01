"""
Utility functions for model operations including saving, loading, and inference.
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from collections import OrderedDict

def save_model(model, filepath, model_type=None, class_names=None, model_variant=None):
    """
    Save the model and metadata.
    
    Args:
        model (nn.Module): Model to save
        filepath (str): Path to save the model
        model_type (str, optional): Type of model architecture
        class_names (list, optional): List of class names
        model_variant (str, optional): Specific model variant
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Handle DataParallel models
    if isinstance(model, nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    
    # Prepare data to save
    save_data = {
        'state_dict': state_dict
    }
    
    # Add metadata if provided
    if model_type:
        save_data['model_type'] = model_type
    if class_names:
        save_data['class_names'] = class_names
    if model_variant:
        save_data['model_variant'] = model_variant
    
    # Save model
    torch.save(save_data, filepath)
    print(f"Model saved to {filepath}")
    
    # Save class names separately as JSON for easier loading in other contexts
    if class_names:
        class_map = {i: name for i, name in enumerate(class_names)}
        json_path = os.path.join(os.path.dirname(filepath), 'class_names.json')
        with open(json_path, 'w') as f:
            json.dump(class_map, f, indent=4)
        print(f"Class names saved to {json_path}")

def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, 
                   train_acc, val_acc, filepath):
    """
    Save a training checkpoint.
    
    Args:
        model (nn.Module): Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler state
        epoch (int): Current epoch
        train_loss (float): Training loss
        val_loss (float): Validation loss
        train_acc (float): Training accuracy
        val_acc (float): Validation accuracy
        filepath (str): Path to save the checkpoint
    """
    # Handle DataParallel models
    if isinstance(model, nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    
    # Create checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc
    }
    
    # Save checkpoint
    torch.save(checkpoint, filepath)

def load_model(filepath, model=None):
    """
    Load a saved model.
    
    Args:
        filepath (str): Path to the saved model
        model (nn.Module, optional): Model instance to load weights into
        
    Returns:
        tuple: (model, model_metadata)
    """
    # Check if file exists
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No model found at {filepath}")
    
    # Load the saved data
    saved_data = torch.load(filepath, map_location=torch.device('cpu'))
    
    # Extract state dict and metadata
    state_dict = saved_data['state_dict']
    metadata = {k: v for k, v in saved_data.items() if k != 'state_dict'}
    
    # If model is provided, load weights into it
    if model is not None:
        # Handle inconsistent keys (e.g., from DataParallel)
        if any(key.startswith('module.') for key in state_dict.keys()):
            # Remove 'module.' prefix
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                new_key = key[7:] if key.startswith('module.') else key
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        
        # Load state dict
        model.load_state_dict(state_dict)
        model.eval()
        return model, metadata
    else:
        # If model is not provided, just return the state dict and metadata
        return state_dict, metadata

def load_class_names(model_dir):
    """
    Load class names from a model directory.
    
    Args:
        model_dir (str): Directory containing the model and class_names.json
        
    Returns:
        list: List of class names
    """
    json_path = os.path.join(model_dir, 'class_names.json')
    if os.path.isfile(json_path):
        with open(json_path, 'r') as f:
            class_map = json.load(f)
        # Convert keys to integers and sort by key
        class_map = {int(k): v for k, v in class_map.items()}
        return [class_map[i] for i in sorted(class_map.keys())]
    else:
        raise FileNotFoundError(f"No class names found at {json_path}")

def get_top_predictions(outputs, class_names, top_k=3):
    """
    Get top-k predictions from model outputs.
    
    Args:
        outputs (torch.Tensor): Model output logits
        class_names (list): List of class names
        top_k (int): Number of top predictions to return
        
    Returns:
        list: List of (class_name, probability) tuples
    """
    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    # Get top-k probabilities and indices
    top_probs, top_indices = torch.topk(probabilities, top_k)
    
    # Convert to lists
    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]
    
    # Create result list
    results = []
    for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
        results.append({
            'rank': i + 1,
            'class_name': class_names[idx],
            'probability': float(prob)
        })
    
    return results

def preprocess_image_for_model(image, is_vit=False, model_input_size=224):
    """
    Preprocess an image file for model inference.
    
    Args:
        image: PIL.Image or path to image file
        is_vit (bool): Whether the model is a Vision Transformer
        model_input_size (int): Model input size
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    from torchvision import transforms
    
    # Load image if path is provided
    if isinstance(image, str):
        try:
            image = Image.open(image).convert('RGB')
        except Exception as e:
            raise ValueError(f"Could not open image file: {e}")
    
    # Standard preprocessing for CNN models
    preprocess = transforms.Compose([
        transforms.Resize(int(model_input_size * 1.14)),  # Resize slightly larger
        transforms.CenterCrop(model_input_size),  # Then center crop
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
    ])
    
    # Preprocess image
    image_tensor = preprocess(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def predict_plant_species(model, image, class_names, device=None, top_k=3, is_vit=False):
    """
    Predict plant species from an image.
    
    Args:
        model (nn.Module): Trained model
        image: PIL.Image or path to image file
        class_names (list): List of class names
        device: Device to run inference on
        top_k (int): Number of top predictions to return
        is_vit (bool): Whether the model is a Vision Transformer
        
    Returns:
        list: Top predictions as list of dictionaries
    """
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set model to evaluation mode
    model.eval()
    model = model.to(device)
    
    # Preprocess image
    input_tensor = preprocess_image_for_model(image, is_vit=is_vit)
    input_tensor = input_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        
        # Handle ViT output format
        if hasattr(output, 'logits'):
            output = output.logits
    
    # Get top predictions
    predictions = get_top_predictions(output, class_names, top_k=top_k)
    
    return predictions

def export_model_for_web(model, class_names, output_dir, model_type=None, model_variant=None):
    """
    Export model for web deployment, including TorchScript serialization.
    
    Args:
        model (nn.Module): Model to export
        class_names (list): List of class names
        output_dir (str): Directory to save exported model
        model_type (str, optional): Type of model architecture
        model_variant (str, optional): Specific model variant
        
    Returns:
        str: Path to exported model
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create a dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    try:
        # Try to convert model to TorchScript
        scripted_model = torch.jit.trace(model, dummy_input)
        
        # Save TorchScript model
        torch_script_path = os.path.join(output_dir, 'model_scripted.pt')
        scripted_model.save(torch_script_path)
        print(f"TorchScript model saved to {torch_script_path}")
        
        # Also save normal PyTorch model for fallback
        pytorch_path = os.path.join(output_dir, 'model.pth')
        save_model(
            model, pytorch_path, 
            model_type=model_type,
            class_names=class_names,
            model_variant=model_variant
        )
        
        # Save class names
        class_map = {i: name for i, name in enumerate(class_names)}
        json_path = os.path.join(output_dir, 'class_names.json')
        with open(json_path, 'w') as f:
            json.dump(class_map, f, indent=4)
        
        # Save model metadata
        metadata = {
            'num_classes': len(class_names),
            'input_size': 224,
            'model_type': model_type,
            'model_variant': model_variant,
            'normalize_mean': [0.485, 0.456, 0.406],
            'normalize_std': [0.229, 0.224, 0.225]
        }
        
        meta_path = os.path.join(output_dir, 'model_metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        return output_dir
        
    except Exception as e:
        print(f"Failed to export TorchScript model: {e}")
        print("Falling back to standard PyTorch export")
        
        # Save just the PyTorch model
        pytorch_path = os.path.join(output_dir, 'model.pth')
        save_model(
            model, pytorch_path, 
            model_type=model_type,
            class_names=class_names,
            model_variant=model_variant
        )
        
        # Save class names and metadata as above
        class_map = {i: name for i, name in enumerate(class_names)}
        json_path = os.path.join(output_dir, 'class_names.json')
        with open(json_path, 'w') as f:
            json.dump(class_map, f, indent=4)
        
        metadata = {
            'num_classes': len(class_names),
            'input_size': 224,
            'model_type': model_type,
            'model_variant': model_variant,
            'normalize_mean': [0.485, 0.456, 0.406],
            'normalize_std': [0.229, 0.224, 0.225]
        }
        
        meta_path = os.path.join(output_dir, 'model_metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        return output_dir

def convert_onnx(model, output_path, input_size=224):
    """
    Convert PyTorch model to ONNX format.
    
    Args:
        model (nn.Module): PyTorch model
        output_path (str): Path to save ONNX model
        input_size (int): Input image size
        
    Returns:
        str: Path to exported ONNX model
    """
    try:
        import onnx
        import onnxruntime
        
        # Set model to evaluation mode
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, input_size, input_size)
        
        # Export to ONNX
        torch.onnx.export(
            model,                      # Model being exported
            dummy_input,                # Model input
            output_path,                # Output file
            export_params=True,         # Store trained parameter weights inside model
            opset_version=12,           # ONNX version to use
            do_constant_folding=True,   # Optimization: fold constant ops
            input_names=['input'],      # Input names
            output_names=['output'],    # Output names
            dynamic_axes={              # Dynamic axes
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify the exported model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        # Test with ONNX Runtime
        ort_session = onnxruntime.InferenceSession(output_path)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_session.run(None, ort_inputs)
        
        print(f"ONNX model saved to {output_path}")
        return output_path
        
    except ImportError:
        print("ONNX export failed: missing onnx or onnxruntime packages")
        return None
    except Exception as e:
        print(f"ONNX export failed: {e}")
        return None

if __name__ == "__main__":
    """Test utility functions with a sample model."""
    import torch.nn as nn
    
    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self, num_classes=10):
            super(TestModel, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = nn.Sequential(
                nn.Linear(32 * 7 * 7, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
    
    # Create instance and test save/load
    test_model = TestModel(num_classes=5)
    test_classes = ['Class1', 'Class2', 'Class3', 'Class4', 'Class5']
    
    # Test directory
    test_dir = 'test_model_utils'
    os.makedirs(test_dir, exist_ok=True)
    
    # Test save_model
    save_model(test_model, f'{test_dir}/test_model.pth', 
               model_type='test', class_names=test_classes)
    
    # Test load_model
    loaded_model, metadata = load_model(f'{test_dir}/test_model.pth', TestModel(num_classes=5))
    
    # Test export_model_for_web
    export_model_for_web(test_model, test_classes, f'{test_dir}/web_export', 
                         model_type='test')
    
    print("Model utility tests completed!")
