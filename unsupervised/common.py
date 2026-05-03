import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def image_paths(image_dir):
    root = Path(image_dir)
    return sorted([*root.glob("*.png"), *root.glob("*.jpg"), *root.glob("*.jpeg")])


def xyxy_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-9)


def nms_xyxy(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order) > 0:
        i = int(order[0])
        keep.append(i)
        rest = []
        for j in order[1:]:
            if xyxy_iou(boxes[i], boxes[int(j)]) <= iou_threshold:
                rest.append(j)
        order = np.array(rest, dtype=int)
    return keep


def draw_detections(image_path, detections, out_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy_px"]
        label = f"{det['class']} {det['confidence']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        draw.text((x1, y1), label, fill=(255, 255, 0))
    img.save(out_path)


def write_detection_outputs(out_dir, detections_by_frame, metadata):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    flat = [det for rows in detections_by_frame.values() for det in rows]
    (out / "detections_by_frame.json").write_text(json.dumps(detections_by_frame, indent=2))
    (out / "detections_flat.json").write_text(json.dumps(flat, indent=2))
    metadata = dict(metadata)
    metadata["num_detections"] = len(flat)
    metadata["num_frames"] = len(detections_by_frame)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))

