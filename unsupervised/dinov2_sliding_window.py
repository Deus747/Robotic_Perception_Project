import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from common import draw_detections, image_paths, nms_xyxy, write_detection_outputs
from dinov2_utils import DINOv2Embedder, cosine, crop_xyxy


def parse_args():
    parser = argparse.ArgumentParser(description="DINOv2 exemplar sliding-window detector.")
    parser.add_argument("--images", required=True)
    parser.add_argument("--reference-image", required=True)
    parser.add_argument("--reference-box", nargs=4, type=float, required=True, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--entity", default="unknown_object")
    parser.add_argument("--window", type=int, default=224)
    parser.add_argument("--overlap", type=float, default=0.75)
    parser.add_argument("--scales", nargs="*", type=float, default=[0.60, 0.75, 0.90, 1.0, 1.10, 1.25, 1.50])
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--nms-iou", type=float, default=0.15)
    return parser.parse_args()


def sliding_boxes(width, height, base_window, overlap, scales):
    for scale in scales:
        win = max(16, int(round(base_window * scale)))
        stride = max(1, int(round(win * (1.0 - overlap))))
        for y in range(0, max(1, height - win + 1), stride):
            for x in range(0, max(1, width - win + 1), stride):
                yield [x, y, min(width, x + win), min(height, y + win)]


def main():
    args = parse_args()
    out = Path(args.out)
    vis = out / "visualizations"
    vis.mkdir(parents=True, exist_ok=True)

    embedder = DINOv2Embedder()
    ref_image = Image.open(args.reference_image).convert("RGB")
    ref_vec = embedder.embed_pil(crop_xyxy(ref_image, args.reference_box))

    detections_by_frame = {}
    for img_path in tqdm(image_paths(args.images)):
        img = Image.open(img_path).convert("RGB")
        scored = []
        for box in sliding_boxes(img.width, img.height, args.window, args.overlap, args.scales):
            vec = embedder.embed_pil(crop_xyxy(img, box))
            score = cosine(ref_vec, vec)
            if score >= args.threshold:
                scored.append((score, box))
        scored = sorted(scored, reverse=True, key=lambda x: x[0])[: args.top_k]
        boxes = [b for _, b in scored]
        scores = [s for s, _ in scored]
        keep = nms_xyxy(boxes, scores, args.nms_iou)
        dets = []
        for rank, idx in enumerate(keep):
            box = boxes[idx]
            dets.append(
                {
                    "frame": img_path.name,
                    "detection_index": rank,
                    "class": args.entity,
                    "raw_class": "dinov2_exemplar",
                    "class_id": 0,
                    "confidence": float(scores[idx]),
                    "bbox_xyxy_px": [float(v) for v in box],
                    "center_px": [float((box[0] + box[2]) / 2.0), float((box[1] + box[3]) / 2.0)],
                }
            )
        detections_by_frame[img_path.name] = dets
        draw_detections(img_path, dets, vis / f"{img_path.name}.png")

    write_detection_outputs(
        out,
        detections_by_frame,
        {
            "method": "dinov2_sliding_window",
            "reference_image": args.reference_image,
            "reference_box": args.reference_box,
            "window": args.window,
            "overlap": args.overlap,
            "scales": args.scales,
            "threshold": args.threshold,
        },
    )


if __name__ == "__main__":
    main()

