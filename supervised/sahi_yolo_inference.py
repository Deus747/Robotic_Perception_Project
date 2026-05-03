import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO with SAHI sliced inference and export reconstruction-ready detections.")
    parser.add_argument("--weights", required=True, help="Path to trained YOLO .pt weights.")
    parser.add_argument("--images", required=True, help="Directory containing PNG/JPG frames.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--slice-size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-class", action="append", default=[], help="Class name to skip; repeatable.")
    return parser.parse_args()


def draw_preview(image_path, detections, out_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy_px"]
        label = f"{det['class']} {det['confidence']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        draw.text((x1, y1), label, fill=(255, 255, 0))
    img.save(out_path)


def main():
    args = parse_args()
    image_dir = Path(args.images)
    out_dir = Path(args.out)
    vis_dir = out_dir / "visualizations_2d"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(exist_ok=True)

    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=args.weights,
        confidence_threshold=args.conf,
        device=args.device,
    )
    skip = set(args.skip_class)
    image_paths = sorted([*image_dir.glob("*.png"), *image_dir.glob("*.jpg"), *image_dir.glob("*.jpeg")])

    by_frame = {}
    flat = []
    for image_path in image_paths:
        result = get_sliced_prediction(
            str(image_path),
            model,
            slice_height=args.slice_size,
            slice_width=args.slice_size,
            overlap_height_ratio=args.overlap,
            overlap_width_ratio=args.overlap,
            perform_standard_pred=True,
            postprocess_type="NMM",
            verbose=0,
        )
        frame_dets = []
        for idx, pred in enumerate(result.object_prediction_list):
            cls = pred.category.name
            if cls in skip:
                continue
            b = pred.bbox
            item = {
                "frame": image_path.name,
                "detection_index": idx,
                "class": cls,
                "raw_class": cls,
                "class_id": int(pred.category.id),
                "confidence": float(pred.score.value),
                "bbox_xyxy_px": [float(b.minx), float(b.miny), float(b.maxx), float(b.maxy)],
                "center_px": [float((b.minx + b.maxx) / 2.0), float((b.miny + b.maxy) / 2.0)],
            }
            frame_dets.append(item)
            flat.append(item)
        by_frame[image_path.name] = frame_dets
        draw_preview(image_path, frame_dets, vis_dir / f"{image_path.name}.png")

    class_counts = {}
    for det in flat:
        class_counts[det["class"]] = class_counts.get(det["class"], 0) + 1

    (out_dir / "detections_by_frame.json").write_text(json.dumps(by_frame, indent=2))
    (out_dir / "detections_flat.json").write_text(json.dumps(flat, indent=2))
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "weights": str(args.weights),
                "images": str(image_dir),
                "num_frames": len(image_paths),
                "num_detections": len(flat),
                "class_counts": class_counts,
                "sahi": {"slice_size": args.slice_size, "overlap": args.overlap, "confidence": args.conf},
            },
            indent=2,
        )
    )
    print(f"Wrote {len(flat)} detections to {out_dir}")


if __name__ == "__main__":
    main()

