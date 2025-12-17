from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# --- Simple "decision layer" for waste handling (can be refined later) ---
INSTRUCTIONS = {
    "plastic": "Bilas cepat, keringkan, lalu buang ke Recycle Plastik.",
    "glass": "Masukkan ke Recycle Kaca. Hati-hati pecah.",
    "metal": "Kosongkan, bilas, lalu buang ke Recycle Logam.",
    "paper": "Pastikan kering (tidak berminyak/basah), buang ke Kertas.",
    "cardboard": "Lipat/pipihkan dan pastikan kering, buang ke Kardus.",
    "trash": "Buang ke Residu/Tempat Sampah Umum (tidak bisa recycle).",
}

@dataclass(frozen=True)
class PredictConfig:
    image_size: int = 224
    confidence_threshold: float = 0.60     # below this -> needs_review
    margin_threshold: float = 0.15         # if top1-top2 small -> needs_review
    topk: int = 2


def _build_infer_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])


def load_labels(labels_path: str | Path = "models/labels.json") -> list[str]:
    import json
    p = Path(labels_path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    return obj["classes"]


def load_model(weights_path: str | Path = "models/model.pth",
               labels_path: str | Path = "models/labels.json",
               device: str | None = None):
    from src.ml.model import create_model

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    labels = load_labels(labels_path)
    model = create_model("resnet18", num_classes=len(labels))
    state = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, labels, device


@torch.no_grad()
def predict_pil_ui(img: Image.Image,
                   model,
                   labels: list[str],
                   device: str,
                   cfg: PredictConfig = PredictConfig()) -> Dict[str, Any]:
    tf = _build_infer_transform(cfg.image_size)
    x = tf(img.convert("RGB")).unsqueeze(0).to(device)

    logits = model(x)
    probs = F.softmax(logits, dim=1).squeeze(0)  # tensor [C]
    probs_cpu = probs.detach().cpu()

    # top-k
    topk_vals, topk_idx = torch.topk(probs_cpu, k=min(cfg.topk, len(labels)))
    top: List[Dict[str, Any]] = []
    for v, i in zip(topk_vals.tolist(), topk_idx.tolist()):
        top.append({"label": labels[int(i)], "confidence": float(v)})

    label1 = top[0]["label"]
    conf1 = top[0]["confidence"]
    conf2 = top[1]["confidence"] if len(top) > 1 else 0.0
    margin = conf1 - conf2

    # heuristic: common confusion pair in TrashNet
    confusable_pairs = {("glass", "plastic"), ("plastic", "glass")}
    top2_pair = (label1, top[1]["label"]) if len(top) > 1 else None
    pair_confusable = top2_pair in confusable_pairs

    needs_review = pair_confusable or (conf1 < cfg.confidence_threshold) or (margin < cfg.margin_threshold)


    return {
        "label": label1,
        "confidence": conf1,
        "top": top,                      # top2 list
        "needs_review": bool(needs_review),
        "margin": float(margin),
        "instruction": INSTRUCTIONS.get(label1, "Buang sesuai kategori yang benar."),
        # full probs (optional, useful for debugging)
        "probs": {labels[i]: float(probs_cpu[i].item()) for i in range(len(labels))}
    }


def predict_path_ui(image_path: str | Path,
                    weights_path: str | Path = "models/model.pth",
                    labels_path: str | Path = "models/labels.json",
                    cfg: PredictConfig = PredictConfig()) -> Dict[str, Any]:
    model, labels, device = load_model(weights_path, labels_path)
    img = Image.open(image_path).convert("RGB")
    return predict_pil_ui(img, model, labels, device, cfg)
