# 单张图推理（含 TenCrop数据增强策略，在src/datasets.py中）
from src.transforms import tta_transforms
from PIL import Image
import torch

def ten_crop_inference(model, img_path, device="cuda"):
    model.eval()
    img = Image.open(img_path).convert("RGB")
    crops = tta_transforms(img).to(device)  # shape: [10, 3, 224, 224]
    with torch.no_grad():
        logits = model(crops)
        probs = torch.softmax(logits, dim=1)
    return probs.mean(0).argmax().item()