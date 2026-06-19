import torch
import torch.nn as nn
import timm
import segmentation_models_pytorch as smp

def get_classifier(num_classes=6, pretrained=True):
    """
    Returns the EfficientNet-B4 classifier model used in the GLOF validation pipeline.
    
    Args:
        num_classes (int): Number of target classification classes (default: 6).
        pretrained (bool): Whether to load pretrained ImageNet weights.
        
    Returns:
        torch.nn.Module: Instantiated timm EfficientNet-B4 model.
    """
    model = timm.create_model("efficientnet_b4", pretrained=pretrained, num_classes=num_classes)
    return model

def get_segmenter(in_channels=3, classes=1, encoder_name="efficientnet-b4", encoder_weights="imagenet"):
    """
    Returns the UNet++ segmenter model with an EfficientNet-B4 encoder used in the GLOF validation pipeline.
    
    Args:
        in_channels (int): Number of input image channels (default: 3).
        classes (int): Number of output segmentation classes/masks (default: 1 for binary).
        encoder_name (str): CNN backbone encoder name (default: "efficientnet-b4").
        encoder_weights (str): Pretrained weights source (default: "imagenet").
        
    Returns:
        torch.nn.Module: Instantiated smp UNet++ model.
    """
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes
    )
    return model
