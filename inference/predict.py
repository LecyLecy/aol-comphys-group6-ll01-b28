from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

def _build_infer_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

def load_labels(labels_path: str | Path = "models/labels.json") -> list[str]:
    import json
    p = Path(labels_path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    return obj["classes"]

def load_model(weights_path: str | Path = "models/model.pth", labels_path: str | Path = "models/labels.json", device: str = "cpu"):
    from src.ml.model import create_model

    labels = load_labels(labels_path)
    model = create_model("resnet18", num_classes=len(labels))
    state = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, labels

@torch.no_grad()
def predict_pil(img: Image.Image, model, labels: list[str], device: str = "cpu", image_size: int = 224) -> Dict[str, Any]:
    tf = _build_infer_transform(image_size=image_size)
    x = tf(img.convert("RGB")).unsqueeze(0).to(device)
    logits = model(x)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_idx = int(probs.argmax())
    conf = float(probs[top_idx])
    label = labels[top_idx]

    return {
        "label": label,
        "confidence": conf,
        "probs": {labels[i]: float(probs[i]) for i in range(len(labels))}
    }

def predict_path(image_path: str | Path, weights_path: str | Path = "models/model.pth", labels_path: str | Path = "models/labels.json") -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, labels = load_model(weights_path, labels_path, device=device)
    img = Image.open(image_path).convert("RGB")
    return predict_pil(img, model, labels, device=device)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inference/predict.py <image_path>")
        raise SystemExit(1)
    out = predict_path(sys.argv[1])
    print(out)
