import timm
import segmentation_models_pytorch as smp

def get_classifier(num_classes=6):
    """EfficientNet-B4 classifier for 6-class glacial lake categorization."""
    return timm.create_model("efficientnet_b4", pretrained=True, num_classes=num_classes)

def get_segmenter():
    """UNet++ with EfficientNet-B4 encoder for binary lake segmentation."""
    return smp.UnetPlusPlus("efficientnet-b4", encoder_weights="imagenet", in_channels=3, classes=1)
