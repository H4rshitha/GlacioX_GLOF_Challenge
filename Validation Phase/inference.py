import os
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model_architecture import get_classifier, get_segmenter

# 6 target training classes
TRAIN_CATEGORIES = [
    "Cloud Cover", "Debris Cover", "Moraine", 
    "Snow Cover", "Terraine Shadow", "Turbidity"
]

def load_models(cls_weights_path, seg_weights_path, device):
    """Loads classification and segmentation models and their weights."""
    print("Loading models...")
    # Classifier (EfficientNet-B4)
    cls_model = get_classifier(num_classes=len(TRAIN_CATEGORIES), pretrained=False)
    if os.path.exists(cls_weights_path):
        cls_model.load_state_dict(torch.load(cls_weights_path, map_location=device))
        print(f"Loaded classification weights from {cls_weights_path}")
    else:
        print(f"Warning: Classification weights not found at {cls_weights_path}")
        
    # Segmenter (UNet++ with EfficientNet-B4)
    seg_model = get_segmenter(in_channels=3, classes=1)
    if os.path.exists(seg_weights_path):
        seg_model.load_state_dict(torch.load(seg_weights_path, map_location=device))
        print(f"Loaded segmentation weights from {seg_weights_path}")
    else:
        print(f"Warning: Segmentation weights not found at {seg_weights_path}")
        
    cls_model.to(device).eval()
    seg_model.to(device).eval()
    return cls_model, seg_model

def run_inference(image_path, cls_model, seg_model, device, output_dir=None):
    """Runs classification and segmentation on a single image and saves the mask."""
    img_orig = cv2.imread(str(image_path))
    if img_orig is None:
        raise ValueError(f"Could not read image from {image_path}")
        
    h_orig, w_orig = img_orig.shape[:2]
    img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    
    # Validation/Inference Preprocessing (Resize to 384x384, Normalize, ToTensorV2)
    transform = A.Compose([
        A.Resize(384, 384),
        A.Normalize(),
        ToTensorV2()
    ])
    
    transformed = transform(image=img_rgb)
    x = transformed["image"].unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Classification prediction
        cls_logits = cls_model(x)
        pred_idx = cls_logits.argmax(dim=1).item()
        pred_class = TRAIN_CATEGORIES[pred_idx]
        
        # Segmentation prediction
        seg_logits = seg_model(x)
        pred_mask = (torch.sigmoid(seg_logits) > 0.5).float()
        
    mask_np = pred_mask[0, 0].cpu().numpy()
    
    # Postprocessing (Resize mask back to original image dimensions)
    mask_resized = cv2.resize(mask_np, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    mask_out = (mask_resized * 255).astype(np.uint8)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stem = Path(image_path).stem
        
        # Save binary mask (0 = background, 255 = lake)
        mask_save_path = output_dir / f"{stem}_mask.png"
        cv2.imwrite(str(mask_save_path), mask_out)
        
        # Save overlay visualization (red mask overlay over original image)
        overlay = img_orig.copy()
        overlay[mask_out == 255] = [0, 0, 255]
        vis_save_path = output_dir / f"{stem}_overlay.png"
        cv2.imwrite(str(vis_save_path), cv2.addWeighted(img_orig, 0.7, overlay, 0.3, 0))
        
        print(f"Inference completed for {Path(image_path).name}:")
        print(f"  - Predicted Class: {pred_class}")
        print(f"  - Binary mask saved to: {mask_save_path}")
        print(f"  - Overlay visualization saved to: {vis_save_path}")
        
    return pred_class, mask_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GLOF Inference Script")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--cls_weights", type=str, default="weights/best_cls_model.pth", help="Path to classification weights (.pth)")
    parser.add_argument("--seg_weights", type=str, default="weights/best_seg_model.pth", help="Path to segmentation weights (.pth)")
    parser.add_argument("--output_dir", type=str, default="inference_output", help="Directory to save output masks")
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        cls_model, seg_model = load_models(args.cls_weights, args.seg_weights, device)
        run_inference(args.image, cls_model, seg_model, device, args.output_dir)
    except Exception as e:
        print(f"Error during inference: {e}")
