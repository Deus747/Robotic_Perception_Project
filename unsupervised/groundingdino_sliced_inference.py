import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from common import draw_detections, image_paths, nms_xyxy, write_detection_outputs


def parse_args():
    parser = argparse.ArgumentParser(description="GroundingDINO manual SAHI-style sliced inference.")
    parser.add_argument("--images", required=True)
    parser.add_argument("--prompts", required=True, help='Comma-separated prompts, e.g. "ethernet socket,power socket".')
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--slice-size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.30)
    parser.add_argument("--box-threshold", type=float, default=0.20)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--nms-iou", type=float, default=0.35)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def tiles(width, height, size, overlap):
    stride = max(1, int(round(size * (1.0 - overlap))))
    y_values = list(range(0, max(1, height - size + 1), stride))
    x_values = list(range(0, max(1, width - size + 1), stride))
    if not y_values or y_values[-1] + size < height:
        y_values.append(max(0, height - size))
    if not x_values or x_values[-1] + size < width:
        x_values.append(max(0, width - size))
    for y in sorted(set(y_values)):
        for x in sorted(set(x_values)):
            yield x, y, min(width, x + size), min(height, y + size)


def main():
    args = parse_args()
    out = Path(args.out)
    vis = out / "visualizations"
    vis.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    text = ". ".join(prompts) + "."

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device).eval()

    detections_by_frame = {}
    for img_path in tqdm(image_paths(args.images)):
        image = Image.open(img_path).convert("RGB")
        candidates = []
        for x1, y1, x2, y2 in tiles(image.width, image.height, args.slice_size, args.overlap):
            tile = image.crop((x1, y1, x2, y2))
            inputs = processor(images=tile, text=text, return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = model(**inputs)
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                target_sizes=[tile.size[::-1]],
            )[0]
            for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
                bx1, by1, bx2, by2 = box.detach().cpu().tolist()
                global_box = [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1]
                candidates.append((float(score), str(label), global_box))

        boxes = [b for _, _, b in candidates]
        scores = [s for s, _, _ in candidates]
        keep = nms_xyxy(boxes, scores, args.nms_iou)
        dets = []
        for rank, idx in enumerate(keep):
            score, label, box = candidates[idx]
            dets.append(
                {
                    "frame": img_path.name,
                    "detection_index": rank,
                    "class": label,
                    "raw_class": label,
                    "class_id": 0,
                    "confidence": score,
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
            "method": "groundingdino_sliced",
            "model": args.model,
            "prompts": prompts,
            "slice_size": args.slice_size,
            "overlap": args.overlap,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
        },
    )


if __name__ == "__main__":
    main()

