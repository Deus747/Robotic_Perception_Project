import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from common import draw_detections, image_paths, nms_xyxy, write_detection_outputs
from dinov2_utils import DINOv2Embedder, cosine, crop_xyxy


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO-World proposal generation ranked by DINOv2 exemplar similarity.")
    parser.add_argument("--images", required=True)
    parser.add_argument("--reference-image", required=True)
    parser.add_argument("--reference-box", nargs=4, type=float, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--entity", default="unknown_object")
    parser.add_argument("--weights", default="yolov8x-worldv2.pt")
    parser.add_argument("--prompts", nargs="*", default=["object", "socket", "connector", "computer part"])
    parser.add_argument("--conf", type=float, default=0.005)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--slice-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.55)
    parser.add_argument("--max-proposals", type=int, default=80)
    parser.add_argument("--similarity-threshold", type=float, default=0.40)
    parser.add_argument("--nms-iou", type=float, default=0.20)
    return parser.parse_args()


def tiles(width, height, size, overlap):
    stride = max(1, int(round(size * (1.0 - overlap))))
    xs = list(range(0, max(1, width - size + 1), stride))
    ys = list(range(0, max(1, height - size + 1), stride))
    if not xs or xs[-1] + size < width:
        xs.append(max(0, width - size))
    if not ys or ys[-1] + size < height:
        ys.append(max(0, height - size))
    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            yield x, y, min(width, x + size), min(height, y + size)


def collect_yoloworld_proposals(model, image, args):
    proposals = []
    for x1, y1, x2, y2 in tiles(image.width, image.height, args.slice_size, args.overlap):
        tile = image.crop((x1, y1, x2, y2))
        result = model.predict(tile, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        if result.boxes is None:
            continue
        for box, conf in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
            bx1, by1, bx2, by2 = box
            w, h = bx2 - bx1, by2 - by1
            if w < 5 or h < 5:
                continue
            proposals.append([float(conf), [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1]])
    return sorted(proposals, reverse=True, key=lambda x: x[0])[: args.max_proposals]


def main():
    args = parse_args()
    out = Path(args.out)
    vis = out / "visualizations"
    vis.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    model.set_classes(args.prompts)
    embedder = DINOv2Embedder()
    ref_img = Image.open(args.reference_image).convert("RGB")
    ref_vec = embedder.embed_pil(crop_xyxy(ref_img, args.reference_box))

    detections_by_frame = {}
    for img_path in tqdm(image_paths(args.images)):
        img = Image.open(img_path).convert("RGB")
        proposals = collect_yoloworld_proposals(model, img, args)
        ranked = []
        for proposal_conf, box in proposals:
            sim = cosine(ref_vec, embedder.embed_pil(crop_xyxy(img, box)))
            combined = 0.90 * sim + 0.10 * proposal_conf
            if sim >= args.similarity_threshold:
                ranked.append((combined, sim, proposal_conf, box))
        ranked = sorted(ranked, reverse=True, key=lambda x: x[0])
        boxes = [row[3] for row in ranked]
        scores = [row[0] for row in ranked]
        keep = nms_xyxy(boxes, scores, args.nms_iou)
        dets = []
        for rank, idx in enumerate(keep):
            combined, sim, proposal_conf, box = ranked[idx]
            dets.append(
                {
                    "frame": img_path.name,
                    "detection_index": rank,
                    "class": args.entity,
                    "raw_class": "yoloworld_dinov2_exemplar",
                    "class_id": 0,
                    "confidence": float(combined),
                    "visual_similarity": float(sim),
                    "proposal_confidence": float(proposal_conf),
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
            "method": "yoloworld_dinov2_exemplar",
            "weights": args.weights,
            "prompts": args.prompts,
            "reference_image": args.reference_image,
            "reference_box": args.reference_box,
            "conf": args.conf,
            "imgsz": args.imgsz,
            "slice_size": args.slice_size,
            "overlap": args.overlap,
        },
    )


if __name__ == "__main__":
    main()

