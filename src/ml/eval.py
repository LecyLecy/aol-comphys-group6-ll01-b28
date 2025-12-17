from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from .config import Config
from .dataset import TrashDataset, build_transforms, Sample
from .model import create_model
from .utils import get_device, load_json, ensure_dir, save_json
from .train import gather_samples, split_samples  # reuse

@torch.no_grad()
def main():
    cfg = Config()
    device = get_device(cfg.device)
    ensure_dir(cfg.metrics_dir)

    # Load labels
    labels_path = Path("models/labels.json")
    labels_obj = load_json(labels_path)
    classes = labels_obj["classes"]

    samples = gather_samples(cfg.data_dir, tuple(classes))
    _, _, test_s = split_samples(samples, cfg.seed, cfg.train_ratio, cfg.val_ratio, cfg.test_ratio)

    _, eval_tf = build_transforms(cfg.image_size)
    test_ds = TrashDataset(test_s, transform=eval_tf)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = create_model(cfg.model_name, num_classes=len(classes)).to(device)
    state = torch.load("models/model.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()

    y_true = []
    y_pred = []

    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(y.numpy().tolist())

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)

    save_json(cfg.metrics_dir / "classification_report.json", report)

    # Plot confusion matrix
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)
    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    out_path = cfg.metrics_dir / "confusion_matrix.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print("Saved:", out_path)
    print("Saved:", cfg.metrics_dir / "classification_report.json")

if __name__ == "__main__":
    main()
