import numpy as np
import cv2

def compute_iou(pred, target):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return intersection / (union + 1e-7)

def mine_lake_patches(img, mask, zoom=2.0):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    patches = []
    for c in contours:
        if cv2.contourArea(c) < 30:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w // 2, y + h // 2
        crop_size = int(max(w, h) * zoom)
        x1, y1 = max(0, cx - crop_size // 2), max(0, cy - crop_size // 2)
        x2, y2 = min(img.shape[1], x1 + crop_size), min(img.shape[0], y1 + crop_size)
        patches.append((img[y1:y2, x1:x2], mask[y1:y2, x1:x2]))
    return patches
