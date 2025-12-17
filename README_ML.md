# TrashNet ML Engineer Template (PyTorch)

This repo section is for the **ML Engineer**: training + evaluation + exporting a model for UI integration.

## Dataset layout expected
Put TrashNet images under:
`data/trashnet/raw/<class_name>/*.jpg`

Class folders (TrashNet standard):
- cardboard
- glass
- metal
- paper
- plastic
- trash

## Quickstart
1) Create venv and install:
```bash
pip install -r requirements.txt
```

2) Train (creates `models/model.pth`, `models/labels.json`, logs to `metrics/`):
```bash
python -m src.ml.train
```

3) Evaluate on test split (if enabled in config):
```bash
python -m src.ml.eval
```

4) Run a single prediction (from file path):
```bash
python -c "from inference.predict import predict_path; print(predict_path('path/to/image.jpg'))"
```

## Outputs (deliverables)
- `models/model.pth` : best weights
- `models/labels.json` : index-to-label order
- `metrics/metrics.json` : accuracy/F1 summary
- `metrics/confusion_matrix.png` : confusion matrix figure

## Notes
- Keep `labels.json` consistent with training class order.
- Inference preprocessing must match training preprocessing (normalize/resize).
