"""
Module for selecting and configuring the vision model for houseplant identification.
Provides functions to create model architectures with various backbones.
"""

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from transformers import ViTForImageClassification, AutoFeatureExtractor
import torchvision.models as models

def create_resnet_model(num_classes, model_size=50, pretrained=True):
    """
    Create a ResNet model for plant classification.
    
    Args:
        num_classes (int): Number of output classes
        model_size (int): ResNet variant (18, 34, 50, 101, 152)
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        nn.Module: Configured ResNet model
    """
    # Select model size
    if model_size == 18:
        model = models.resnet18(pretrained=pretrained)
    elif model_size == 34:
        model = models.resnet34(pretrained=pretrained)
    elif model_size == 50:
        model = models.resnet50(pretrained=pretrained)
    elif model_size == 101:
        model = models.resnet101(pretrained=pretrained)
    elif model_size == 152:
        model = models.resnet152(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported ResNet size: {model_size}")
    
    # Modify final fully connected layer for our number of classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

def create_efficient_net_model(num_classes, model_variant='b0', pretrained=True):
    """
    Create an EfficientNet model for plant classification.
    
    Args:
        num_classes (int): Number of output classes
        model_variant (str): EfficientNet variant (b0 through b7)
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        nn.Module: Configured EfficientNet model
    """
    # Map variant to model function
    if model_variant == 'b0':
        model = models.efficientnet_b0(pretrained=pretrained)
    elif model_variant == 'b1':
        model = models.efficientnet_b1(pretrained=pretrained)
    elif model_variant == 'b2':
        model = models.efficientnet_b2(pretrained=pretrained)
    elif model_variant == 'b3':
        model = models.efficientnet_b3(pretrained=pretrained)
    elif model_variant == 'b4':
        model = models.efficientnet_b4(pretrained=pretrained)
    elif model_variant == 'b5':
        model = models.efficientnet_b5(pretrained=pretrained)
    elif model_variant == 'b6':
        model = models.efficientnet_b6(pretrained=pretrained)
    elif model_variant == 'b7':
        model = models.efficientnet_b7(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported EfficientNet variant: {model_variant}")
    
    # Modify classifier for our number of classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model

def create_vit_model(num_classes, model_variant='base', pretrained=True):
    """
    Create a Vision Transformer (ViT) model for plant classification using HuggingFace.
    
    Args:
        num_classes (int): Number of output classes
        model_variant (str): ViT variant ('base', 'large')
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        nn.Module: Configured ViT model
    """
    # Choose model variant
    if model_variant == 'base':
        model_name = 'google/vit-base-patch16-224'
    elif model_variant == 'large':
        model_name = 'google/vit-large-patch16-224'
    else:
        raise ValueError(f"Unsupported ViT variant: {model_variant}")
    
    # Load pretrained model
    if pretrained:
        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True  # Important when changing the head
        )
    else:
        # Load model config only (without pretrained weights)
        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            from_config=True
        )
    
    return model, AutoFeatureExtractor.from_pretrained(model_name)

def create_mobile_net_model(num_classes, pretrained=True):
    """
    Create a MobileNetV3 model for plant classification.
    More suitable for mobile deployment.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        nn.Module: Configured MobileNetV3 model
    """
    # Load MobileNetV3 small for better efficiency
    model = models.mobilenet_v3_small(pretrained=pretrained)
    
    # Modify classifier for our number of classes
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    
    return model

def get_model(model_type, num_classes, **kwargs):
    """
    Factory function to get a configured model based on type.
    
    Args:
        model_type (str): Type of model ('resnet', 'efficientnet', 'vit', 'mobilenet')
        num_classes (int): Number of output classes
        **kwargs: Additional model-specific parameters
        
    Returns:
        nn.Module: Configured model
    """
    if model_type == 'resnet':
        return create_resnet_model(
            num_classes,
            model_size=kwargs.get('model_size', 50),
            pretrained=kwargs.get('pretrained', True)
        )
    elif model_type == 'efficientnet':
        return create_efficient_net_model(
            num_classes,
            model_variant=kwargs.get('model_variant', 'b0'),
            pretrained=kwargs.get('pretrained', True)
        )
    elif model_type == 'vit':
        return create_vit_model(
            num_classes,
            model_variant=kwargs.get('model_variant', 'base'),
            pretrained=kwargs.get('pretrained', True)
        )
    elif model_type == 'mobilenet':
        return create_mobile_net_model(
            num_classes,
            pretrained=kwargs.get('pretrained', True)
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def get_recommended_model(num_classes, target='web'):
    """
    Get the recommended model based on deployment target.
    
    Args:
        num_classes (int): Number of output classes
        target (str): Deployment target ('web', 'mobile', 'high_accuracy')
        
    Returns:
        nn.Module: Recommended model
    """
    if target == 'mobile':
        # MobileNetV3 is optimized for mobile
        return create_mobile_net_model(num_classes)
    elif target == 'high_accuracy':
        # EfficientNet B3 offers good accuracy/performance tradeoff
        return create_efficient_net_model(num_classes, model_variant='b3')
    else:  # 'web' or default
        # ResNet50 is a good balance for web deployment
        return create_resnet_model(num_classes, model_size=50)

if __name__ == "__main__":
    """Test model creation with a sample number of classes."""
    num_test_classes = 20
    
    # Test ResNet
    resnet = create_resnet_model(num_test_classes)
    print(f"Created ResNet model: {resnet.__class__.__name__}")
    print(f"Output layer: {resnet.fc}")
    
    # Test EfficientNet
    efficientnet = create_efficient_net_model(num_test_classes)
    print(f"Created EfficientNet model: {efficientnet.__class__.__name__}")
    print(f"Output layer: {efficientnet.classifier[1]}")
    
    # Test ViT
    vit, feature_extractor = create_vit_model(num_test_classes)
    print(f"Created ViT model: {vit.__class__.__name__}")
    
    # Test MobileNet
    mobilenet = create_mobile_net_model(num_test_classes)
    print(f"Created MobileNet model: {mobilenet.__class__.__name__}")
    print(f"Output layer: {mobilenet.classifier[3]}")
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        resnet_output = resnet(dummy_input)
        efficientnet_output = efficientnet(dummy_input)
        mobilenet_output = mobilenet(dummy_input)
        
    print(f"ResNet output shape: {resnet_output.shape}")
    print(f"EfficientNet output shape: {efficientnet_output.shape}")
    print(f"MobileNet output shape: {mobilenet_output.shape}")
