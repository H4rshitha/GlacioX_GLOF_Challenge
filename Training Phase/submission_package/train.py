import torch
import timm
import segmentation_models_pytorch as smp

def get_classifier(num_classes=6):
    return timm.create_model("efficientnet_b4", pretrained=True, num_classes=num_classes)

def get_segmenter():
    return smp.UnetPlusPlus("efficientnet-b4", encoder_weights="imagenet", in_channels=3, classes=1)

if __name__ == "__main__":
    print("V14 Training Script - Use Kaggle notebook for full pipeline.")
