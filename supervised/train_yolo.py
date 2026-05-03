import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train supervised YOLO detector for known project objects.")
    parser.add_argument("--data-yaml", required=True, help="YOLO dataset YAML.")
    parser.add_argument("--weights", default="yolo26s.pt", help="Initial YOLO weights, e.g. yolo26s.pt.")
    parser.add_argument("--project", default="runs/yolo_known", help="Ultralytics output project directory.")
    parser.add_argument("--name", default="train", help="Run name.")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--warmup-epochs", type=float, default=5.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if not Path(args.data_yaml).exists():
        raise FileNotFoundError(args.data_yaml)

    model = YOLO(args.weights)
    model.train(
        data=args.data_yaml,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        lr0=args.lr0,
        warmup_epochs=args.warmup_epochs,
        dropout=args.dropout,
        project=args.project,
        name=args.name,
        device=args.device,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()

