"""Fine-tune a COCO-pretrained YOLO nano model on the simulated can dataset."""
import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
from ultralytics import YOLO, settings

settings.update({"sync": False})  # no usage analytics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt", help="yolov8n.pt or yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    (ROOT / "models").mkdir(exist_ok=True)
    data = str(ROOT / "data" / "can" / "data.yaml")
    model = YOLO(str(ROOT / "models" / args.model))
    model.train(
        data=data, epochs=args.epochs, imgsz=640, batch=args.batch, device=device,
        project=str(ROOT / "runs"), name="yolo_can", exist_ok=True, seed=0,
        patience=15, workers=4, close_mosaic=8, hsv_h=0.01, degrees=5.0, plots=True,
    )
    best = ROOT / "models" / "can_detector.pt"
    shutil.copy(ROOT / "runs" / "yolo_can" / "weights" / "best.pt", best)

    m = YOLO(str(best)).val(data=data, split="test", imgsz=640, device=device,
                            project=str(ROOT / "runs"), name="yolo_test", exist_ok=True)
    out = {"base_model": args.model, "precision": m.box.mp, "recall": m.box.mr,
           "mAP50": m.box.map50, "mAP50-95": m.box.map}
    (ROOT / "runs" / "yolo_test_metrics.json").write_text(json.dumps(out, indent=2))
    print("TEST", json.dumps(out))


if __name__ == "__main__":
    main()
