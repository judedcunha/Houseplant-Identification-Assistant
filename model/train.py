"""
Script for training the houseplant identification model.
"""

import os
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Import from our modules
import sys
sys.path.append('..')
from model.model_selection import get_model
from data.preprocess_images import create_dataloaders
from model.model_utils import save_model, load_model, save_checkpoint

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): Model to train
        dataloader (DataLoader): Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Create progress bar
    pbar = tqdm(dataloader, desc="Training")
    
    for inputs, targets in pbar:
        # Move data to device
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        if isinstance(model, nn.DataParallel):
            outputs = model.module(inputs)
        else:
            outputs = model(inputs)
            
        # Handle different model output formats (ViT vs traditional)
        if hasattr(outputs, 'logits'):
            outputs = outputs.logits
        
        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })
    
    # Calculate epoch metrics
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """
    Validate the model on validation data.
    
    Args:
        model (nn.Module): Model to validate
        dataloader (DataLoader): Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        tuple: (average loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validating"):
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            if isinstance(model, nn.DataParallel):
                outputs = model.module(inputs)
            else:
                outputs = model(inputs)
                
            # Handle different model output formats (ViT vs traditional)
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Update metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Store predictions and targets for detailed metrics
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate validation metrics
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc, all_preds, all_targets

def plot_metrics(train_losses, val_losses, train_accs, val_accs, save_path):
    """
    Plot and save training metrics.
    
    Args:
        train_losses (list): Training losses
        val_losses (list): Validation losses
        train_accs (list): Training accuracies
        val_accs (list): Validation accuracies
        save_path (str): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_metrics_report(all_preds, all_targets, class_names, save_dir):
    """
    Save detailed classification metrics.
    
    Args:
        all_preds (list): All predictions
        all_targets (list): All true labels
        class_names (list): List of class names
        save_dir (str): Directory to save reports
    """
    # Generate classification report
    report = classification_report(
        all_targets, all_preds, 
        target_names=class_names, 
        output_dict=True
    )
    
    # Save as JSON
    with open(os.path.join(save_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    # Generate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    
    # Add numbers to cells
    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save plot
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def train_model(model_type, num_epochs, batch_size, learning_rate, weight_decay=1e-4,
                model_variant=None, pretrained=True, output_dir='outputs'):
    """
    Train the plant identification model.
    
    Args:
        model_type (str): Type of model ('resnet', 'efficientnet', 'vit', 'mobilenet')
        num_epochs (int): Number of epochs to train
        batch_size (int): Batch size
        learning_rate (float): Initial learning rate
        weight_decay (float): Weight decay for regularization
        model_variant (str, optional): Specific model variant
        pretrained (bool): Whether to use pretrained weights
        output_dir (str): Directory to save outputs
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader, class_names = create_dataloaders(batch_size=batch_size)
    print(f"Created dataloaders with {len(class_names)} classes")
    
    # Create model
    model_kwargs = {'pretrained': pretrained}
    if model_variant:
        if model_type == 'resnet':
            model_kwargs['model_size'] = int(model_variant)
        else:
            model_kwargs['model_variant'] = model_variant
    
    model = get_model(model_type, len(class_names), **model_kwargs)
    
    # Move model to device
    model = model.to(device)
    
    # Use DataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{model_type}_{model_variant if model_variant else 'default'}"
    output_path = os.path.join(output_dir, f"{model_name}_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    
    # Train the model
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    print(f"Starting training for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, all_preds, all_targets = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(
                model, 
                os.path.join(output_path, 'best_model.pth'),
                model_type=model_type,
                class_names=class_names,
                model_variant=model_variant
            )
            print(f"Saved new best model with validation accuracy: {val_acc:.2f}%")
        
        # Save checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch, train_loss, val_loss,
            train_acc, val_acc, os.path.join(output_path, f'checkpoint_epoch_{epoch+1}.pth')
        )
    
    # Calculate training time
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Plot and save metrics
    plot_metrics(
        train_losses, val_losses, train_accs, val_accs,
        os.path.join(output_path, 'training_metrics.png')
    )
    
    # Final validation for detailed metrics
    _, _, final_preds, final_targets = validate(model, val_loader, criterion, device)
    save_metrics_report(final_preds, final_targets, class_names, output_path)
    
    # Save the final model
    save_model(
        model, 
        os.path.join(output_path, 'final_model.pth'),
        model_type=model_type,
        class_names=class_names,
        model_variant=model_variant
    )
    
    # Save configuration
    config = {
        'model_type': model_type,
        'model_variant': model_variant,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'pretrained': pretrained,
        'best_val_acc': best_val_acc,
        'num_classes': len(class_names),
        'training_time_minutes': total_time/60,
        'timestamp': timestamp
    }
    
    with open(os.path.join(output_path, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"All outputs saved to {output_path}")
    return output_path

def main():
    """Main function to parse arguments and start training."""
    parser = argparse.ArgumentParser(description='Train plant identification model')
    parser.add_argument('--model', type=str, default='mobilenet',
                        choices=['resnet', 'efficientnet', 'vit', 'mobilenet'],
                        help='Model architecture')
    parser.add_argument('--variant', type=str, default=None,
                        help='Model variant (e.g., 50 for ResNet50, b0 for EfficientNet-B0)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Do not use pretrained weights')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Train the model
    train_model(
        model_type=args.model,
        model_variant=args.variant,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        pretrained=not args.no_pretrained,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
