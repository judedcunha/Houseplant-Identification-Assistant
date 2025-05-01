"""
Script for preprocessing plant images for model training.
Includes resizing, normalization, and data augmentation functions.
"""

import os
import json
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Default image size for model input
IMAGE_SIZE = 224

def create_transforms(is_training=True):
    """
    Create image transformation pipelines for training and validation sets.
    
    Args:
        is_training (bool): Whether to create transforms for training (with augmentation)
                           or validation (without augmentation)
    
    Returns:
        transforms.Compose: Transformation pipeline
    """
    if is_training:
        # More aggressive augmentation for training set
        transform = transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225]     # ImageNet stds
            )
        ])
    else:
        # Simpler preprocessing for validation set
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    return transform

class HouseplantDataset(Dataset):
    """Dataset class for houseplant images."""
    
    def __init__(self, data_dir, transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_dir (str): Directory containing plant images organized in class folders
            transform (callable, optional): Transform to apply to images
        """
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(data_dir) 
                              if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Load label mapping if available
        self.label_map = None
        label_map_path = os.path.join(os.path.dirname(os.path.dirname(data_dir)), 
                                     "label_mapping.json")
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r') as f:
                # Convert keys to ints since JSON stores them as strings
                self.label_map = {int(k): v for k, v in json.load(f).items()}
        
        # Collect all image paths and labels
        self.samples = self._make_dataset()
        
    def _make_dataset(self):
        """
        Create a list of (image_path, class_idx) tuples.
        
        Returns:
            list: List of (image_path, class_idx) tuples
        """
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for root, _, files in os.walk(class_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        samples.append((os.path.join(root, file), class_idx))
        
        return samples
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, class_idx)
        """
        image_path, class_idx = self.samples[idx]
        
        # Load and convert image
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms if available
            if self.transform:
                image = self.transform(image)
            
            return image, class_idx
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a placeholder if image loading fails
            if self.transform:
                # Create a black image
                placeholder = torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE))
                return placeholder, class_idx
            else:
                placeholder = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color='black')
                return placeholder, class_idx
    
    def get_class_name(self, class_idx):
        """
        Get the class name for a given class index.
        
        Args:
            class_idx (int): Class index
            
        Returns:
            str: Class name (scientific or common name if available)
        """
        if self.label_map and class_idx in self.label_map:
            return self.label_map[class_idx].get('common_name', 
                                                 self.label_map[class_idx].get('scientific_name'))
        else:
            return self.classes[class_idx]

def create_dataloaders(batch_size=32, num_workers=4):
    """
    Create DataLoader objects for training and validation sets.
    
    Args:
        batch_size (int): Batch size for training
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        tuple: (train_loader, val_loader, class_names)
    """
    # Create transforms
    train_transform = create_transforms(is_training=True)
    val_transform = create_transforms(is_training=False)
    
    # Create datasets
    train_dataset = HouseplantDataset(
        data_dir='data/processed/train',
        transform=train_transform
    )
    
    val_dataset = HouseplantDataset(
        data_dir='data/processed/val',
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Created dataset with {len(train_dataset)} training and "
          f"{len(val_dataset)} validation samples across "
          f"{len(train_dataset.classes)} classes.")
    
    return train_loader, val_loader, train_dataset.classes

def preprocess_single_image(image_path, size=IMAGE_SIZE):
    """
    Preprocess a single image for inference.
    
    Args:
        image_path (str): Path to the image file
        size (int): Target image size
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Create inference transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load and transform image
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image)
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

if __name__ == "__main__":
    """Test the preprocessing functions with a sample image."""
    # If a test image is provided, try preprocessing it
    test_image_path = "data/test_sample.jpg"
    if os.path.exists(test_image_path):
        print(f"Testing preprocessing with image: {test_image_path}")
        tensor = preprocess_single_image(test_image_path)
        if tensor is not None:
            print(f"Successfully preprocessed image to tensor of shape {tensor.shape}")
    
    # Test the dataloaders
    try:
        train_loader, val_loader, classes = create_dataloaders(batch_size=4)
        print(f"Sample classes: {classes[:5]}...")
        
        # Get a batch
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}, Labels: {labels}")
        print("Preprocessing test completed successfully.")
    except Exception as e:
        print(f"Error testing dataloaders: {e}")
