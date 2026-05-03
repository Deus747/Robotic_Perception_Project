import argparse
import json
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert simple bbox annotations to a YOLO dataset skeleton.")
    parser.add_argument("--annotations", required=True, help="Annotation JSON containing image/frame and bbox records.")
    parser.add_argument("--images", required=True, help="Source image directory.")
    parser.add_argument("--out", required=True, help="Output YOLO dataset directory.")
    parser.add_argument("--class-map", required=True, help='JSON map, e.g. {"power_socket": 0, "ethernet_port": 1}.')
    parser.add_argument("--val-ratio", type=float, default=0.2)
    return parser.parse_args()


def yolo_line(box, image_width, image_height, class_id):
    x1, y1, x2, y2 = [float(v) for v in box]
    cx = ((x1 + x2) / 2.0) / image_width
    cy = ((y1 + y2) / 2.0) / image_height
    w = (x2 - x1) / image_width
    h = (y2 - y1) / image_height
    return f"{class_id} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}"


def main():
    from PIL import Image

    args = parse_args()
    ann = json.loads(Path(args.annotations).read_text())
    class_map = json.loads(args.class_map)
    image_dir = Path(args.images)
    out = Path(args.out)

    records = ann if isinstance(ann, list) else ann.get("annotations", ann.get("records", []))
    by_image = {}
    for row in records:
        frame = row.get("frame") or row.get("image") or row.get("file_name")
        label = row.get("class") or row.get("label") or row.get("category")
        box = row.get("bbox_xyxy_px") or row.get("bbox_px") or row.get("box")
        if frame and label in class_map and box:
            by_image.setdefault(frame, []).append((label, box))

    frames = sorted(by_image)
    split_at = int(round(len(frames) * (1.0 - args.val_ratio)))
    splits = {"train": frames[:split_at], "val": frames[split_at:]}

    for split, split_frames in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for frame in split_frames:
            src = image_dir / frame
            if not src.exists():
                continue
            dst = out / "images" / split / frame
            shutil.copy2(src, dst)
            with Image.open(src) as img:
                width, height = img.size
            lines = [yolo_line(box, width, height, class_map[label]) for label, box in by_image[frame]]
            (out / "labels" / split / f"{Path(frame).stem}.txt").write_text("\n".join(lines) + "\n")

    names = {v: k for k, v in class_map.items()}
    yaml_text = [
        f"path: {out.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for idx in sorted(names):
        yaml_text.append(f"  {idx}: {names[idx]}")
    (out / "data.yaml").write_text("\n".join(yaml_text) + "\n")
    print(f"Wrote YOLO dataset to {out}")


if __name__ == "__main__":
    main()

