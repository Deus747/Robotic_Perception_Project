import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_DETECTIONS = ROOT / "yolo26_9class_results" / "sahi_results_9class" / "detections.json"
SOURCE_VIS = ROOT / "yolo26_9class_results" / "sahi_results_9class"
SOURCE_WEIGHTS = ROOT / "yolo26_9class_results" / "base_weights_9class" / "weights" / "best.pt"
OUTPUT_DIR = ROOT / "obb_reconstruction_inputs" / "yolo26_9class_sahi"

CLASS_NAME_MAP = {
    "power_socket": "power_socket",
    "ethernet_port": "ethernet_port",
    "vga_port": "vga_port",
    "bottle": "bottle",
    "key": "key",
    "glass": "glass",
    "purifier": "purifier",
    "ps": "ps",
    "sticker": "sticker",
}


def main():
    if not SOURCE_DETECTIONS.exists():
        raise FileNotFoundError(f"Missing detections file: {SOURCE_DETECTIONS}")
    if not SOURCE_WEIGHTS.exists():
        raise FileNotFoundError(f"Missing weight file: {SOURCE_WEIGHTS}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vis_out = OUTPUT_DIR / "visualizations"
    vis_out.mkdir(exist_ok=True)

    raw = json.loads(SOURCE_DETECTIONS.read_text())
    frames = sorted(raw.keys())

    normalized = {}
    flat = []
    class_counts = {}

    for frame_name in frames:
        dets = []
        for i, det in enumerate(raw[frame_name]):
            cls = det["class"]
            normalized_name = CLASS_NAME_MAP.get(cls, cls)
            x1, y1, x2, y2 = [float(v) for v in det["bbox_px"]]
            cx, cy = [float(v) for v in det["center_px"]]
            item = {
                "frame": frame_name,
                "detection_index": i,
                "class": normalized_name,
                "raw_class": cls,
                "class_id": int(det["class_id"]),
                "confidence": float(det["conf"]),
                "bbox_xyxy_px": [x1, y1, x2, y2],
                "center_px": [cx, cy],
                "width_px": float(x2 - x1),
                "height_px": float(y2 - y1),
            }
            dets.append(item)
            flat.append(item)
            class_counts[normalized_name] = class_counts.get(normalized_name, 0) + 1
        normalized[frame_name] = dets

    metadata = {
        "source_weight": str(SOURCE_WEIGHTS),
        "source_detections": str(SOURCE_DETECTIONS),
        "num_frames": len(frames),
        "num_detections": len(flat),
        "frames": frames,
        "class_counts": class_counts,
        "notes": [
            "These detections come from YOLO26 9-class SAHI sliced inference already present in the workspace.",
            "BBox format is pixel-space xyxy on the original Data/*.png images.",
            "Use these detections as 2D observations for downstream 3D OBB reconstruction.",
        ],
    }

    (OUTPUT_DIR / "detections_by_frame.json").write_text(json.dumps(normalized, indent=2))
    (OUTPUT_DIR / "detections_flat.json").write_text(json.dumps(flat, indent=2))
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    for png in SOURCE_VIS.glob("*.png"):
        if png.name == "detections.json":
            continue
        shutil.copy2(png, vis_out / png.name)

    print(f"Wrote normalized reconstruction inputs to: {OUTPUT_DIR}")
    print(f"Frames: {len(frames)}")
    print(f"Detections: {len(flat)}")
    print("Class counts:")
    for k in sorted(class_counts):
        print(f"  {k}: {class_counts[k]}")


if __name__ == "__main__":
    main()
