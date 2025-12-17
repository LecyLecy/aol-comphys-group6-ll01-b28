from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    # Paths
    data_dir: Path = Path("data/trashnet/raw")
    output_dir: Path = Path("models")
    metrics_dir: Path = Path("metrics")

    # Classes (TrashNet standard)
    classes: tuple[str, ...] = ("cardboard", "glass", "metal", "paper", "plastic", "trash")

    # Splits
    seed: int = 42
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10

    # Image/model
    image_size: int = 224
    model_name: str = "resnet18"  # keep simple and fast

    # Train
    batch_size: int = 32
    num_epochs: int = 12
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2

    # Early stopping
    patience: int = 3

    # Device
    device: str = "cuda"  # will fallback to cpu if unavailable

    # Threshold for UI (optional use)
    confidence_threshold: float = 0.60
