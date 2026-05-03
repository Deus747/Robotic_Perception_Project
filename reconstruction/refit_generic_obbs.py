import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "obb_reconstruction_inputs" / "yolo26_9class_sahi_rerun_no_purifier"
INPUT_DETECTIONS = INPUT_DIR / "detections_flat.json"
INTRINSIC_PATH = ROOT / "intrinsic.json"
POSES_PATH = ROOT / "poses.json"

OUT_DIR = ROOT / "obb_reconstruction_inputs" / "yolo26_9class_generic_refit"
VIS3D_OVERLAY_DIR = OUT_DIR / "visualizations_3d_overlays"


GENERIC_HYPOTHESES = [
    {"name": "compact_center_identity", "anchor": "center", "rotation": "identity", "extent0": [0.05, 0.05, 0.05], "prior_weight": 0.22},
    {"name": "compact_center_camera", "anchor": "center", "rotation": "camera_facing", "extent0": [0.05, 0.05, 0.05], "prior_weight": 0.22},
    {"name": "upright_center", "anchor": "center", "rotation": "upright", "extent0": [0.05, 0.05, 0.20], "prior_weight": 0.28},
    {"name": "upright_bottom", "anchor": "bottom_center", "rotation": "upright", "extent0": [0.05, 0.05, 0.20], "prior_weight": 0.28},
    {"name": "flat_center", "anchor": "center", "rotation": "planar_facing", "extent0": [0.08, 0.006, 0.05], "prior_weight": 0.30},
    {"name": "flat_bottom", "anchor": "bottom_center", "rotation": "planar_facing", "extent0": [0.08, 0.006, 0.05], "prior_weight": 0.30},
    {"name": "elongated_center", "anchor": "center", "rotation": "camera_facing", "extent0": [0.10, 0.03, 0.03], "prior_weight": 0.24},
]


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


def observation_anchor(obs, mode):
    x1, y1, x2, y2 = obs["bbox_xyxy_px"]
    if mode == "bottom_center":
        return [(x1 + x2) / 2.0, y2]
    if mode == "top_center":
        return [(x1 + x2) / 2.0, y1]
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


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


def average_camera_center(poses, frames):
    centers = []
    for frame_name in frames:
        T_wc = pose_matrix_for_frame(poses, frame_name)
        centers.append(T_wc[:3, 3])
    return np.mean(np.asarray(centers, dtype=np.float64), axis=0)


def make_camera_facing_rotation(center0, poses, frames):
    avg_cam = average_camera_center(poses, frames)
    z_axis = avg_cam - center0
    z_axis = z_axis / np.linalg.norm(z_axis)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(up, z_axis)) > 0.9:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def make_upright_rotation(center0, poses, frames):
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    avg_cam = average_camera_center(poses, frames)
    facing = avg_cam - center0
    facing = facing - np.dot(facing, world_up) * world_up
    if np.linalg.norm(facing) < 1e-6:
        facing = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = facing / np.linalg.norm(facing)
    z_axis = world_up
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def make_planar_facing_rotation(center0, poses, frames):
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    avg_cam = average_camera_center(poses, frames)
    normal = avg_cam - center0
    normal = normal / np.linalg.norm(normal)
    x_axis = np.cross(world_up, normal)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    return np.column_stack([x_axis, y_axis, normal])


def rotation_from_mode(mode, center0, poses, frames):
    if mode == "identity":
        return np.eye(3, dtype=np.float64)
    if mode == "upright":
        return make_upright_rotation(center0, poses, frames)
    if mode == "planar_facing":
        return make_planar_facing_rotation(center0, poses, frames)
    return make_camera_facing_rotation(center0, poses, frames)


def observation_residuals(observations, K, poses, center, extent, rotation):
    residuals = []
    for obs in observations:
        pred = project_obb_box(K, poses, center, extent, rotation, obs["frame"], use_inverse_pose=True)
        if pred is None:
            residuals.append(1e6)
            continue
        target = np.array(obs["bbox_xyxy_px"], dtype=np.float64)
        scale = max(10.0, math.sqrt(max(1.0, (target[2] - target[0]) * (target[3] - target[1]))))
        residuals.append(float(np.linalg.norm((pred - target) / scale)))
    return residuals


def fit_hypothesis(observations, K, poses, hyp):
    frames = [obs["frame"] for obs in observations]
    ray_origins = []
    ray_dirs = []
    for obs in observations:
        uv = observation_anchor(obs, hyp["anchor"])
        C, d = camera_center_and_ray_world(K, poses, obs["frame"], uv, use_inverse_pose=True)
        ray_origins.append(C)
        ray_dirs.append(d)
    center0 = triangulate_point_from_rays(ray_origins, ray_dirs)
    R0 = rotation_from_mode(hyp["rotation"], center0, poses, frames)
    e0 = np.array(hyp["extent0"], dtype=np.float64)
    prior_weight = float(hyp["prior_weight"])

    def residual(params, obs_set):
        center = params[:3]
        extent = np.exp(params[3:6])
        res = []
        for obs in obs_set:
            pred = project_obb_box(K, poses, center, extent, R0, obs["frame"], use_inverse_pose=True)
            if pred is None:
                res.extend([1000, 1000, 1000, 1000])
                continue
            target = np.array(obs["bbox_xyxy_px"], dtype=np.float64)
            scale = max(10.0, math.sqrt(max(1.0, (target[2] - target[0]) * (target[3] - target[1]))))
            res.extend(((pred - target) / scale).tolist())
        res.extend(((np.log(extent) - np.log(e0)) * prior_weight).tolist())
        return np.asarray(res, dtype=np.float64)

    x0 = np.r_[center0, np.log(e0)]
    first = least_squares(lambda p: residual(p, observations), x0, loss="soft_l1", f_scale=1.0, max_nfev=2000)
    center1 = first.x[:3]
    extent1 = np.exp(first.x[3:6])
    per_obs = observation_residuals(observations, K, poses, center1, extent1, R0)

    refined_obs = observations
    if len(observations) >= 5:
        order = np.argsort(per_obs)
        keep_n = max(3, int(math.ceil(0.8 * len(observations))))
        keep_idx = sorted(order[:keep_n].tolist())
        refined_obs = [observations[i] for i in keep_idx]

    second = least_squares(lambda p: residual(p, refined_obs), first.x, loss="soft_l1", f_scale=1.0, max_nfev=2000)
    center2 = second.x[:3]
    extent2 = np.exp(second.x[3:6])
    per_obs2 = observation_residuals(observations, K, poses, center2, extent2, R0)
    mean_err = float(np.mean(per_obs2))

    return {
        "hypothesis": hyp["name"],
        "anchor_mode": hyp["anchor"],
        "rotation_mode": hyp["rotation"],
        "center": center2.tolist(),
        "extent": extent2.tolist(),
        "rotation": R0.tolist(),
        "cost": float(second.cost),
        "mean_observation_error": mean_err,
        "kept_observations": len(refined_obs),
        "num_observations": len(observations),
        "per_observation_error": per_obs2,
    }


def choose_best_fit(observations, K, poses):
    candidates = [fit_hypothesis(observations, K, poses, hyp) for hyp in GENERIC_HYPOTHESES]
    candidates = sorted(candidates, key=lambda x: (x["mean_observation_error"], x["cost"]))
    return candidates[0], candidates


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
        draw.text((int(min_xy[0]), int(min_xy[1])), cls, fill=color)
    img.save(out_path)


def plot_3d_scene(obb_results, poses, out_path):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    frame_ids = [319, 333, 353, 359, 365, 371, 390, 400, 426, 449, 461, 468, 471, 496, 515, 531]
    cams = []
    for k in [str(i) for i in frame_ids]:
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
    ax.set_title("Generic Multi-Hypothesis 3D OBB Refit")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    if not INPUT_DETECTIONS.exists():
        raise FileNotFoundError(INPUT_DETECTIONS)
    detections = json.loads(INPUT_DETECTIONS.read_text())
    intrinsic = json.loads(INTRINSIC_PATH.read_text())
    poses = json.loads(POSES_PATH.read_text())
    K = np.array(intrinsic["camera_matrix"], dtype=np.float64)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS3D_OVERLAY_DIR.mkdir(exist_ok=True)

    grouped, dropped_duplicates = group_single_observation_per_frame(detections)

    best = {}
    all_candidates = {}
    for cls, obs in grouped.items():
        frames = {o["frame"] for o in obs}
        if len(frames) < 2:
            continue
        best_fit, candidates = choose_best_fit(obs, K, poses)
        best[cls] = best_fit
        all_candidates[cls] = candidates

    (OUT_DIR / "obb_estimates.json").write_text(json.dumps(best, indent=2))
    (OUT_DIR / "all_hypotheses.json").write_text(json.dumps(all_candidates, indent=2))
    (OUT_DIR / "metadata.json").write_text(json.dumps({
        "source_detections": str(INPUT_DETECTIONS),
        "notes": [
            "Generic refit over existing 2D detections only.",
            "Tried multiple anchor modes and generic shape/rotation hypotheses.",
            "Used inverse pose convention.",
            "Dropped worst 20 percent of observations when there were at least 5 views, then refit.",
            "For reconstruction, duplicate detections of the same class in the same frame are collapsed to the largest box.",
        ],
        "dropped_duplicate_observations": [
            {
                "frame": det["frame"],
                "class": det["class"],
                "confidence": det.get("confidence"),
                "bbox_xyxy_px": det["bbox_xyxy_px"],
            }
            for det in dropped_duplicates
        ],
        "hypotheses": GENERIC_HYPOTHESES,
    }, indent=2))

    image_paths = sorted((ROOT / "Data").glob("*.png"))
    for img_path in image_paths:
        draw_projected_obb_overlays(
            img_path,
            img_path.name,
            best,
            K,
            poses,
            VIS3D_OVERLAY_DIR / f"{img_path.name}.png",
        )

    plot_3d_scene(best, poses, OUT_DIR / "reconstruction_3d.png")
    print(f"Wrote generic refit outputs to {OUT_DIR}")
    print("Chosen hypotheses:")
    for cls, fit in sorted(best.items()):
        print(f"  {cls}: {fit['hypothesis']} | mean_err={fit['mean_observation_error']:.4f} | cost={fit['cost']:.4f}")


if __name__ == "__main__":
    main()
