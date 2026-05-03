import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data"
INTRINSIC_PATH = ROOT / "intrinsic.json"
POSES_PATH = ROOT / "poses.json"
WEIGHT_PATH = ROOT / "yolo26_9class_results" / "base_weights_9class" / "weights" / "best.pt"
OUT_DIR = ROOT / "obb_reconstruction_inputs" / "yolo26_9class_sahi_rerun_no_purifier"
VIS2D_DIR = OUT_DIR / "visualizations_2d"
VIS3D_OVERLAY_DIR = OUT_DIR / "visualizations_3d_overlays"

SKIP_CLASSES = {"purifier"}
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

VGA_OBB = {
    "center": [0.2704921202927293, 0.2261220732082181, 0.8349008829378597],
    "extent": [0.03537766175069747, 0.011822199241650923, 0.0061316691090621735],
    "rotation": [
        [-0.004004375172752437, 0.9672545151126772, -0.25377680739897346],
        [0.01584254528462312, 0.25380835519540434, 0.9671247761234889],
        [0.9998664804554559, -0.00014774012094266402, -0.016340117333610394],
    ],
}

DEFAULT_EXTENTS = {
    "power_socket": [0.03, 0.03, 0.03],
    "ethernet_port": [0.025, 0.02, 0.02],
    "vga_port": VGA_OBB["extent"],
    "bottle": [0.07, 0.07, 0.22],
    "key": [0.03, 0.01, 0.06],
    "glass": [0.08, 0.08, 0.12],
    "ps": [0.05, 0.02, 0.10],
    "sticker": [0.06, 0.002, 0.04],
}


def detection_area(det):
    x1, y1, x2, y2 = det["bbox_xyxy_px"]
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def group_single_observation_per_frame(detections):
    grouped_by_key = {}
    dropped = []
    for det in detections:
        key = (det["class"], det["frame"])
        prev = grouped_by_key.get(key)
        if prev is None:
            grouped_by_key[key] = det
            continue
        if detection_area(det) > detection_area(prev):
            dropped.append(prev)
            grouped_by_key[key] = det
        else:
            dropped.append(det)

    grouped = {}
    for det in grouped_by_key.values():
        grouped.setdefault(det["class"], []).append(det)
    for obs in grouped.values():
        obs.sort(key=lambda x: x["frame"])
    return grouped, dropped


def frame_index_from_name(name: str) -> str:
    digits = "".join(ch for ch in name if ch.isdigit())
    return str(int(digits))


def pose_matrix_for_frame(poses, frame_name):
    return np.array(poses[frame_index_from_name(frame_name)], dtype=np.float64)


def camera_center_and_ray_world(K, poses, frame_name, uv, use_inverse_pose=True):
    T = pose_matrix_for_frame(poses, frame_name)
    T_wc = T if use_inverse_pose else np.linalg.inv(T)
    C = T_wc[:3, 3]
    R_wc = T_wc[:3, :3]
    pix = np.array([uv[0], uv[1], 1.0], dtype=np.float64)
    d_cam = np.linalg.inv(K) @ pix
    d_cam = d_cam / np.linalg.norm(d_cam)
    d_world = R_wc @ d_cam
    d_world = d_world / np.linalg.norm(d_world)
    return C, d_world


def triangulate_point_from_rays(ray_origins, ray_dirs):
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    I = np.eye(3)
    for C, d in zip(ray_origins, ray_dirs):
        d = d / np.linalg.norm(d)
        P = I - np.outer(d, d)
        A += P
        b += P @ C
    return np.linalg.solve(A + 1e-9 * I, b)


def obb_corners(center, extent, rotation):
    center = np.asarray(center, dtype=np.float64)
    extent = np.asarray(extent, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    signs = np.array([[sx, sy, sz] for sx in [-0.5, 0.5] for sy in [-0.5, 0.5] for sz in [-0.5, 0.5]], dtype=np.float64)
    return center + (signs * extent) @ rotation.T


def project_points(K, points_world, T_world_to_cam):
    pts = np.asarray(points_world, dtype=np.float64)
    homog = np.c_[pts, np.ones(len(pts))]
    cam = (T_world_to_cam @ homog.T).T[:, :3]
    uvw = (K @ cam.T).T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-9)
    return uv, cam[:, 2]


def project_obb_box(K, poses, center, extent, rotation, frame_name, use_inverse_pose=True):
    T = pose_matrix_for_frame(poses, frame_name)
    if use_inverse_pose:
        T = np.linalg.inv(T)
    uv, z = project_points(K, obb_corners(center, extent, rotation), T)
    if np.any(z <= 0):
        return None
    return np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()], dtype=np.float64)


def make_upright_rotation(center0, poses, frames, world_up=np.array([0.0, 0.0, 1.0], dtype=np.float64)):
    cam_centers = []
    for frame_name in frames:
        T_wc = pose_matrix_for_frame(poses, frame_name)
        cam_centers.append(T_wc[:3, 3])
    cam_centers = np.array(cam_centers, dtype=np.float64)
    avg_cam = cam_centers.mean(axis=0)
    facing = avg_cam - center0
    facing = facing - np.dot(facing, world_up) * world_up
    if np.linalg.norm(facing) < 1e-6:
        facing = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = facing / np.linalg.norm(facing)
    z_axis = world_up / np.linalg.norm(world_up)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def make_planar_camera_facing_rotation(center0, poses, frames, world_up=np.array([0.0, 0.0, 1.0], dtype=np.float64)):
    cam_centers = []
    for frame_name in frames:
        T_wc = pose_matrix_for_frame(poses, frame_name)
        cam_centers.append(T_wc[:3, 3])
    cam_centers = np.array(cam_centers, dtype=np.float64)
    avg_cam = cam_centers.mean(axis=0)
    normal = avg_cam - center0
    normal = normal / np.linalg.norm(normal)
    x_axis = np.cross(world_up, normal)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    return np.column_stack([x_axis, y_axis, normal])


def rotation_for_class(class_name, center0, K, poses, observations):
    if class_name in {"power_socket", "ethernet_port", "vga_port"}:
        return np.array(VGA_OBB["rotation"], dtype=np.float64)
    frames = [obs["frame"] for obs in observations]
    if class_name == "bottle":
        return make_upright_rotation(center0, poses, frames)
    if class_name == "glass":
        return make_upright_rotation(center0, poses, frames)
    if class_name == "sticker":
        return make_planar_camera_facing_rotation(center0, poses, frames)
    if class_name == "key":
        return make_planar_camera_facing_rotation(center0, poses, frames)
    T_wc = pose_matrix_for_frame(poses, observations[0]["frame"])
    C0 = T_wc[:3, 3]
    z_axis = C0 - center0
    z_axis = z_axis / np.linalg.norm(z_axis)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(up, z_axis)) > 0.9:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def fit_simple_obb(class_name, observations, K, poses):
    ray_origins = []
    ray_dirs = []
    for obs in observations:
        x1, y1, x2, y2 = obs["bbox_xyxy_px"]
        uv = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
        C, d = camera_center_and_ray_world(K, poses, obs["frame"], uv, use_inverse_pose=True)
        ray_origins.append(C)
        ray_dirs.append(d)
    center0 = triangulate_point_from_rays(ray_origins, ray_dirs)
    R0 = rotation_for_class(class_name, center0, K, poses, observations)
    e0 = np.array(DEFAULT_EXTENTS.get(class_name, [0.03, 0.03, 0.03]), dtype=np.float64)
    prior_weight = 0.20
    if class_name in {"power_socket", "ethernet_port", "vga_port"}:
        prior_weight = 0.35
    elif class_name in {"bottle", "glass"}:
        prior_weight = 0.30
    elif class_name in {"sticker"}:
        prior_weight = 0.40

    def residual(params):
        center = params[:3]
        extent = np.exp(params[3:6])
        res = []
        for obs in observations:
            pred = project_obb_box(K, poses, center, extent, R0, obs["frame"], use_inverse_pose=True)
            if pred is None:
                res.extend([1000, 1000, 1000, 1000])
                continue
            target = np.array(obs["bbox_xyxy_px"], dtype=np.float64)
            scale = max(10.0, math.sqrt(max(1.0, (target[2] - target[0]) * (target[3] - target[1]))))
            res.extend(((pred - target) / scale).tolist())
        res.extend(((np.log(extent) - np.log(e0)) * prior_weight).tolist())
        return np.array(res, dtype=np.float64)

    result = least_squares(residual, np.r_[center0, np.log(e0)], loss="soft_l1", f_scale=1.0, max_nfev=2000)
    return {
        "class": class_name,
        "num_observations": len(observations),
        "center": result.x[:3].tolist(),
        "extent": np.exp(result.x[3:6]).tolist(),
        "rotation": R0.tolist(),
        "cost": float(result.cost),
    }


def draw_2d_visual(image_path, detections, out_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy_px"]
        label = f"{det['class']} {det['confidence']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        draw.text((x1, y1), label, fill=(255, 255, 0))
    img.save(out_path)


def draw_projected_obb_overlays(image_path, frame_name, obb_results, K, poses, out_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    T = pose_matrix_for_frame(poses, frame_name)
    T = np.linalg.inv(T)
    colors = [
        (255, 0, 0),
        (0, 128, 255),
        (0, 180, 0),
        (255, 140, 0),
        (180, 0, 180),
        (120, 80, 40),
        (0, 180, 180),
        (220, 0, 120),
    ]
    edges = [(0,1),(0,2),(0,4),(3,1),(3,2),(3,7),(5,1),(5,4),(5,7),(6,2),(6,4),(6,7)]

    for idx, (cls, obb) in enumerate(sorted(obb_results.items())):
        color = colors[idx % len(colors)]
        corners = obb_corners(np.array(obb["center"]), np.array(obb["extent"]), np.array(obb["rotation"]))
        uv, z = project_points(K, corners, T)
        if np.any(z <= 0):
            continue
        uv = uv.astype(int)
        for a, b in edges:
            draw.line([tuple(uv[a]), tuple(uv[b])], fill=color, width=3)
        min_xy = uv.min(axis=0)
        label_pos = (int(min_xy[0]), int(min_xy[1]))
        draw.text(label_pos, cls, fill=color)

    img.save(out_path)


def plot_3d_scene(obb_results, poses, out_path):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    pose_keys = sorted(poses.keys(), key=lambda x: int(x))
    sampled = [k for k in pose_keys if int(k) in [319, 333, 353, 359, 365, 371, 390, 400, 426, 449, 461, 468, 471, 496, 515, 531]]
    cams = []
    for k in sampled:
        T_wc = np.array(poses[k], dtype=np.float64)
        cams.append(T_wc[:3, 3])
    cams = np.array(cams)
    ax.scatter(cams[:, 0], cams[:, 1], cams[:, 2], c="black", s=25, label="camera centers")

    colors = ["red", "blue", "green", "orange", "purple", "brown", "cyan", "magenta"]
    edges = [(0,1),(0,2),(0,4),(3,1),(3,2),(3,7),(5,1),(5,4),(5,7),(6,2),(6,4),(6,7)]
    for idx, (cls, obb) in enumerate(sorted(obb_results.items())):
        color = colors[idx % len(colors)]
        center = np.array(obb["center"], dtype=np.float64)
        ax.scatter([center[0]], [center[1]], [center[2]], c=color, s=60, label=cls)
        corners = obb_corners(center, np.array(obb["extent"]), np.array(obb["rotation"]))
        for a, b in edges:
            seg = corners[[a, b]]
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, linewidth=1.5)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Approximate 3D OBB Reconstructions from YOLO26 + SAHI Detections")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(DATA_DIR)
    if not WEIGHT_PATH.exists():
        raise FileNotFoundError(WEIGHT_PATH)

    intrinsic = json.loads(INTRINSIC_PATH.read_text())
    poses = json.loads(POSES_PATH.read_text())
    K = np.array(intrinsic["camera_matrix"], dtype=np.float64)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS2D_DIR.mkdir(exist_ok=True)
    VIS3D_OVERLAY_DIR.mkdir(exist_ok=True)

    from torch import cuda
    device = "cuda" if cuda.is_available() else "cpu"
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(WEIGHT_PATH),
        confidence_threshold=0.50,
        device=device,
    )

    image_paths = sorted(DATA_DIR.glob("*.png"))
    detections_by_frame = {}
    flat = []

    for img_path in image_paths:
        result = get_sliced_prediction(
            str(img_path),
            model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.3,
            overlap_width_ratio=0.3,
            perform_standard_pred=True,
            postprocess_type="NMM",
            verbose=0,
        )
        frame_dets = []
        for idx, pred in enumerate(result.object_prediction_list):
            cls = pred.category.name
            cls_norm = CLASS_NAME_MAP.get(cls, cls)
            if cls_norm in SKIP_CLASSES:
                continue
            b = pred.bbox
            item = {
                "frame": img_path.name,
                "detection_index": idx,
                "class": cls_norm,
                "raw_class": cls,
                "class_id": int(pred.category.id),
                "confidence": float(pred.score.value),
                "bbox_xyxy_px": [float(b.minx), float(b.miny), float(b.maxx), float(b.maxy)],
                "center_px": [float((b.minx + b.maxx) / 2.0), float((b.miny + b.maxy) / 2.0)],
            }
            frame_dets.append(item)
            flat.append(item)
        detections_by_frame[img_path.name] = frame_dets
        draw_2d_visual(img_path, frame_dets, VIS2D_DIR / f"{img_path.name}.png")

    grouped, dropped_duplicates = group_single_observation_per_frame(flat)

    obb_results = {}
    for cls, obs in grouped.items():
        frames = {o["frame"] for o in obs}
        if len(frames) < 2:
            continue
        obb_results[cls] = fit_simple_obb(cls, obs, K, poses)

    metadata = {
        "source_weight": str(WEIGHT_PATH),
        "device": device,
        "skipped_classes": sorted(SKIP_CLASSES),
        "num_frames": len(image_paths),
        "num_detections": len(flat),
        "raw_class_counts": {
            cls: sum(1 for det in flat if det["class"] == cls)
            for cls in sorted({det["class"] for det in flat})
        },
        "reconstruction_class_counts": {cls: len(obs) for cls, obs in sorted(grouped.items())},
        "dropped_duplicate_observations": [
            {
                "frame": det["frame"],
                "class": det["class"],
                "confidence": det.get("confidence"),
                "bbox_xyxy_px": det["bbox_xyxy_px"],
            }
            for det in dropped_duplicates
        ],
        "notes": [
            "Purifier was skipped during rerun because SAHI was generating multiple instances for that large object.",
            "3D reconstructions here are simple multi-view OBB fits from 2D detections only.",
            "For reconstruction, duplicate detections of the same class in the same frame are collapsed to the largest box.",
            "Pose convention uses the inverse option, matching prior projection checks on this dataset.",
        ],
    }

    (OUT_DIR / "detections_by_frame.json").write_text(json.dumps(detections_by_frame, indent=2))
    (OUT_DIR / "detections_flat.json").write_text(json.dumps(flat, indent=2))
    (OUT_DIR / "obb_estimates.json").write_text(json.dumps(obb_results, indent=2))
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    plot_3d_scene(obb_results, poses, OUT_DIR / "reconstruction_3d.png")
    for img_path in image_paths:
        draw_projected_obb_overlays(
            img_path,
            img_path.name,
            obb_results,
            K,
            poses,
            VIS3D_OVERLAY_DIR / f"{img_path.name}.png",
        )

    print(f"Wrote outputs to {OUT_DIR}")
    print(f"Frames: {len(image_paths)}")
    print(f"Detections: {len(flat)}")
    print("Classes reconstructed:", sorted(obb_results.keys()))


if __name__ == "__main__":
    main()
