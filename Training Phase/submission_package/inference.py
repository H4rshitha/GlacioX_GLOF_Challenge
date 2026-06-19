import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 384
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

val_transform = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize(), ToTensorV2()])

def predict_class(model, image_path):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    aug = val_transform(image=img)
    x = aug["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
    return logits.argmax(dim=1).item()

def predict_mask(model, image_path):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    aug = val_transform(image=img)
    x = aug["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mask = (torch.sigmoid(model(x)) > 0.5).cpu().numpy()[0, 0]
    return (mask * 255).astype(np.uint8)

if __name__ == "__main__":
    print("V14 Inference Script")
