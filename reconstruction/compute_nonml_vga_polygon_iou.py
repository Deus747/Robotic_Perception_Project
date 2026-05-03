import itertools
import json
import re
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
NONML_DIR = ROOT / "obb_reconstruction_inputs" / "yolo26_9class_nonml_geometric_refit"
ESTIMATE_PATH = NONML_DIR / "sample_answers.json"
SAMPLE_PATH = ROOT / "sample_answers.json"
POSES_PATH = ROOT / "poses.json"
INTRINSIC_PATH = ROOT / "intrinsic.json"
IMAGE_DIR = ROOT / "Data"
OUTPUT_PATH = NONML_DIR / "vga_polygon_iou_report.json"


def frame_number_from_name(name: str) -> int:
    match = re.search(r"(\d+)", name)
    if not match:
        raise ValueError(f"Could not parse frame number from {name}")
    return int(match.group(1))


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError("Zero-length vector")
    return vec / norm


def build_pose_lookup(data: dict) -> dict[int, dict]:
    lookup = {}
    for raw_key, mat in data.items():
        frame_id = int(raw_key)
        c2w = np.array(mat, dtype=np.float64)
        if c2w.shape != (4, 4):
            continue
        lookup[frame_id] = {
            "frame_id": frame_id,
            "name": f"frame_{frame_id:06d}.png",
            "c2w": c2w,
            "w2c": np.linalg.inv(c2w),
            "position": c2w[:3, 3].copy(),
        }
    return lookup


def extract_entity(path: Path, entity_name: str) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.find("\n  },\n  {", start)
        if start < 0 or end < 0:
            raise
        return json.loads(text[start : end + len("\n  }")])

    for item in data:
        if item.get("entity") == entity_name:
            return item
    raise KeyError(f"Entity {entity_name} not found in {path}")


def box_corners(center: np.ndarray, extent: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    half = extent / 2.0
    corners = []
    for sx, sy, sz in itertools.product([-1.0, 1.0], repeat=3):
        local = np.array([sx * half[0], sy * half[1], sz * half[2]], dtype=np.float64)
        corners.append(center + rotation @ local)
    return np.array(corners, dtype=np.float64)


def project_points_to_image(points_world: np.ndarray, K: np.ndarray, w2c: np.ndarray) -> np.ndarray | None:
    points_2d = []
    for point_world in points_world:
        point_h = np.append(point_world, 1.0)
        point_cam = w2c @ point_h
        if point_cam[2] <= 1e-8:
            return None
        pixel = K @ point_cam[:3]
        points_2d.append([pixel[0] / pixel[2], pixel[1] / pixel[2]])
    return np.array(points_2d, dtype=np.float32)


def convex_hull_polygon(points_2d: np.ndarray) -> np.ndarray:
    hull = cv2.convexHull(points_2d.reshape(-1, 1, 2))
    return hull.reshape(-1, 2)


def polygon_iou(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    poly_a = poly_a.astype(np.float32)
    poly_b = poly_b.astype(np.float32)
    area_a = abs(float(cv2.contourArea(poly_a)))
    area_b = abs(float(cv2.contourArea(poly_b)))
    inter_area, inter_poly = cv2.intersectConvexConvex(poly_a, poly_b)
    if inter_poly is None:
        inter_area = 0.0
    union_area = area_a + area_b - float(inter_area)
    if union_area <= 1e-8:
        return 0.0
    return float(inter_area / union_area)


def face_corners_from_front(center: np.ndarray, extent: np.ndarray, rotation: np.ndarray, camera_mean: np.ndarray) -> np.ndarray:
    thickness_axis_idx = min(range(3), key=lambda idx: extent[idx])
    width_idx, height_idx = [idx for idx in range(3) if idx != thickness_axis_idx]
    width_axis = rotation[:, width_idx]
    height_axis = rotation[:, height_idx]
    normal = rotation[:, thickness_axis_idx]
    if np.dot(normal, camera_mean - center) < 0:
        normal = -normal
    face_center = center + normal * (extent[thickness_axis_idx] / 2.0)
    half_w = extent[width_idx] / 2.0
    half_h = extent[height_idx] / 2.0
    return np.array(
        [
            face_center - half_w * width_axis - half_h * height_axis,
            face_center + half_w * width_axis - half_h * height_axis,
            face_center + half_w * width_axis + half_h * height_axis,
            face_center - half_w * width_axis + half_h * height_axis,
        ],
        dtype=np.float64,
    )


def project_face_to_local_2d(face_world: np.ndarray, origin: np.ndarray, axis_u: np.ndarray, axis_v: np.ndarray) -> np.ndarray:
    rel = face_world - origin
    return np.array(
        [[float(np.dot(p, axis_u)), float(np.dot(p, axis_v))] for p in rel],
        dtype=np.float32,
    )


def main() -> None:
    est = extract_entity(ESTIMATE_PATH, "vga_socket")["obb"]
    sample = extract_entity(SAMPLE_PATH, "vga_socket")["obb"]

    est_center = np.array(est["center"], dtype=np.float64)
    est_extent = np.array(est["extent"], dtype=np.float64)
    est_rotation = np.array(est["rotation"], dtype=np.float64)

    sample_center = np.array(sample["center"], dtype=np.float64)
    sample_extent = np.array(sample["extent"], dtype=np.float64)
    sample_rotation = np.array(sample["rotation"], dtype=np.float64)

    intrinsic = json.loads(INTRINSIC_PATH.read_text(encoding="utf-8"))
    poses = json.loads(POSES_PATH.read_text(encoding="utf-8"))
    K = np.array(intrinsic["camera_matrix"], dtype=np.float64)
    pose_lookup = build_pose_lookup(poses)

    camera_positions = np.array([item["position"] for item in pose_lookup.values()], dtype=np.float64)
    camera_mean = np.mean(camera_positions, axis=0)

    est_box = box_corners(est_center, est_extent, est_rotation)
    sample_box = box_corners(sample_center, sample_extent, sample_rotation)
    est_face = face_corners_from_front(est_center, est_extent, est_rotation, camera_mean)
    sample_face = face_corners_from_front(sample_center, sample_extent, sample_rotation, camera_mean)

    sample_thickness_axis_idx = min(range(3), key=lambda idx: sample_extent[idx])
    sample_width_idx, sample_height_idx = [idx for idx in range(3) if idx != sample_thickness_axis_idx]
    sample_axis_u = normalize(sample_rotation[:, sample_width_idx])
    sample_axis_v = normalize(sample_rotation[:, sample_height_idx])

    est_face_local = convex_hull_polygon(
        project_face_to_local_2d(est_face, sample_face.mean(axis=0), sample_axis_u, sample_axis_v)
    )
    sample_face_local = convex_hull_polygon(
        project_face_to_local_2d(sample_face, sample_face.mean(axis=0), sample_axis_u, sample_axis_v)
    )
    face_plane_polygon_iou = polygon_iou(est_face_local, sample_face_local)

    xy_est = convex_hull_polygon(est_box[:, :2].astype(np.float32))
    xy_sample = convex_hull_polygon(sample_box[:, :2].astype(np.float32))
    top_down_xy_hull_iou = polygon_iou(xy_est, xy_sample)

    full_projected_ious = []
    face_projected_ious = []
    per_frame = []

    for image_path in sorted(IMAGE_DIR.glob("*.png")):
        frame_id = frame_number_from_name(image_path.name)
        pose = pose_lookup.get(frame_id)
        if pose is None:
            continue

        est_box_2d = project_points_to_image(est_box, K, pose["w2c"])
        sample_box_2d = project_points_to_image(sample_box, K, pose["w2c"])
        est_face_2d = project_points_to_image(est_face, K, pose["w2c"])
        sample_face_2d = project_points_to_image(sample_face, K, pose["w2c"])
        if any(item is None for item in (est_box_2d, sample_box_2d, est_face_2d, sample_face_2d)):
            continue

        full_iou = polygon_iou(convex_hull_polygon(est_box_2d), convex_hull_polygon(sample_box_2d))
        face_iou = polygon_iou(convex_hull_polygon(est_face_2d), convex_hull_polygon(sample_face_2d))
        full_projected_ious.append(full_iou)
        face_projected_ious.append(face_iou)
        per_frame.append(
            {
                "frame": image_path.name,
                "full_projected_hull_iou": full_iou,
                "front_face_polygon_iou": face_iou,
            }
        )

    report = {
        "estimated_vga_obb": {"entity": "vga_socket", "obb": est},
        "sample_vga_obb": {"entity": "vga_socket", "obb": sample},
        "polygon_iou_metrics": {
            "front_face_plane_polygon_iou": face_plane_polygon_iou,
            "top_down_xy_hull_iou": top_down_xy_hull_iou,
            "num_validation_frames": len(per_frame),
            "mean_full_projected_hull_iou": float(np.mean(full_projected_ious)) if full_projected_ious else 0.0,
            "mean_front_face_projected_polygon_iou": float(np.mean(face_projected_ious)) if face_projected_ious else 0.0,
            "per_frame": per_frame,
        },
    }

    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(OUTPUT_PATH)
    print(json.dumps(report["polygon_iou_metrics"], indent=2))


if __name__ == "__main__":
    main()
