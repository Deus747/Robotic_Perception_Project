import argparse
import base64
import json
import math
import re
from io import BytesIO
from pathlib import Path

import gradio as gr
import numpy as np
from gradio_image_prompter import ImagePrompter
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import least_squares

from trex2_api import TRex2APIWrapper


VGA_OBB = {
    "center": [0.2704921202927293, 0.2261220732082181, 0.8349008829378597],
    "extent": [0.03537766175069747, 0.011822199241650923, 0.0061316691090621735],
    "rotation": [
        [-0.004004375172752437, 0.9672545151126772, -0.25377680739897346],
        [0.01584254528462312, 0.25380835519540434, 0.9671247761234889],
        [0.9998664804554559, -0.00014774012094266402, -0.016340117333610394],
    ],
}

WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)

GEOMETRIC_HYPOTHESES = [
    {"name": "center_compact", "mode": "center_compact", "extent0": [0.05, 0.05, 0.05], "prior_weight": 0.22},
    {"name": "tb_axis", "mode": "top_bottom_axis", "extent0": [0.05, 0.05, 0.12], "prior_weight": 0.24},
    {"name": "tb_axis_support", "mode": "top_bottom_axis_support", "extent0": [0.05, 0.05, 0.12], "prior_weight": 0.24},
    {"name": "upright_support", "mode": "upright_support", "extent0": [0.05, 0.05, 0.16], "prior_weight": 0.26},
    {"name": "planar_facing", "mode": "planar_facing", "extent0": [0.08, 0.01, 0.05], "prior_weight": 0.28},
]

trex2 = None
DEFAULT_DATASET_DIR = str(Path(__file__).resolve().parents[1])


def arg_parse():
    parser = argparse.ArgumentParser(description="Dataset-aware T-Rex2 app")
    parser.add_argument("--trex2_api_token", type=str, required=True)
    parser.add_argument("--server_name", type=str, default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7862)
    return parser.parse_args()


def read_rgb(path: str):
    import cv2

    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def pil_from_path(path: str):
    return Image.open(path).convert("RGB")


def parse_visual_prompt(points):
    boxes = []
    for point in points or []:
        if point[2] == 2 and point[-1] == 3:
            x1, y1, _, x2, y2, _ = point
            boxes.append([float(x1), float(y1), float(x2), float(y2)])
    return boxes


def draw_boxes(image_pil: Image.Image, boxes, scores=None, color=(0, 255, 0), labels=None):
    draw = ImageDraw.Draw(image_pil)
    font_size = max(18, min(36, image_pil.width // 40))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        line_width = max(3, image_pil.width // 500)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        txt = ""
        if labels is not None:
            txt += str(labels[i])
        if scores is not None:
            txt += f" {scores[i]:.2f}" if txt else f"{scores[i]:.2f}"
        if txt:
            bbox = draw.textbbox((x1, y1), txt, font=font)
            padded = [bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2]
            draw.rectangle(padded, fill=color)
            draw.text((x1, y1), txt, fill="white", font=font)
    return image_pil


def compact_error_text(err, limit=220):
    text = str(err).replace("\r", " ").replace("\n", " ").strip()
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text


def build_gallery(dataset_dir: str):
    root = Path(dataset_dir)
    data_dir = root / "Data"
    if not data_dir.exists():
        raise gr.Error(f"Missing Data directory under {dataset_dir}")
    image_paths = sorted(str(p) for p in data_dir.glob("*.png"))
    if not image_paths:
        raise gr.Error("No PNG files found in Data/")
    intrinsic = json.load(open(root / "intrinsic.json"))
    poses = json.load(open(root / "poses.json"))
    state = {
        "dataset_dir": str(root),
        "image_paths": image_paths,
        "intrinsic": intrinsic,
        "poses": poses,
        "reference_indices": [],
        "reference_boxes": [],
        "detections": {},
        "confirmed": {},
        "obb_result": None,
        "obb_results": [],
    }
    gallery = [(p, Path(p).name) for p in image_paths]
    summary = f"Loaded {len(image_paths)} frames from {dataset_dir}"
    return state, gallery, summary, gr.update(choices=[], value=None)


def select_reference(evt: gr.SelectData, state):
    if state is None:
        raise gr.Error("Load a dataset first")
    idx = evt.index
    if isinstance(idx, tuple):
        idx = idx[0]
    refs = list(state["reference_indices"])
    if idx in refs:
        refs.remove(idx)
    else:
        if len(refs) >= 2:
            raise gr.Error("This app supports exactly 2 reference frame slots")
        refs.append(idx)
    state["reference_indices"] = refs
    names = [Path(state["image_paths"][i]).name for i in refs]
    selected_names = [Path(state["image_paths"][i]).name for i in refs]
    return (
        state,
        "\n".join(names) if names else "No reference frames selected",
        gr.update(choices=selected_names, value=selected_names[-1] if selected_names else None),
    )


def remove_selected_reference(state, reference_frame):
    if state is None:
        raise gr.Error("Load a dataset first")
    if not reference_frame:
        return state, "No reference frame selected", gr.update(choices=[], value=None)
    refs = [
        idx
        for idx in state.get("reference_indices", [])
        if Path(state["image_paths"][idx]).name != reference_frame
    ]
    state["reference_indices"] = refs
    selected_names = [Path(state["image_paths"][i]).name for i in refs]
    return (
        state,
        "\n".join(selected_names) if selected_names else "No reference frames selected",
        gr.update(choices=selected_names, value=selected_names[-1] if selected_names else None),
    )


def populate_reference_prompters(state):
    if state is None:
        raise gr.Error("Load a dataset first")
    updates = []
    refs = state["reference_indices"]
    for slot in range(2):
        if slot < len(refs):
            img = read_rgb(state["image_paths"][refs[slot]])
            updates.append({"image": img, "points": []})
        else:
            updates.append(None)
    return updates


def build_prompts_from_prompters(state, prompters):
    refs = state["reference_indices"]
    prompts = []
    reference_boxes = []
    for slot, idx in enumerate(refs):
        payload = prompters[slot]
        if payload is None:
            continue
        boxes = parse_visual_prompt(payload.get("points", []))
        if not boxes:
            raise gr.Error(f"No box drawn for reference slot {slot + 1}")
        ref_image = Image.fromarray(payload["image"])
        interactions = [dict(type="rect", category_id=1, rect=box) for box in boxes]
        prompts.append(dict(image=ref_image, interactions=interactions))
        frame_name = Path(state["image_paths"][idx]).name
        for box in boxes:
            reference_boxes.append(
                {
                    "frame": frame_name,
                    "box": [float(v) for v in box],
                    "score": 1.0,
                    "source": "reference",
                }
            )
    if not prompts:
        raise gr.Error("No valid reference prompts found")
    return prompts, reference_boxes


def run_trex_detection(state, ref1, ref2, visual_threshold, max_target_frames, show_all_candidates):
    if state is None:
        raise gr.Error("Load a dataset first")
    prompts, reference_boxes = build_prompts_from_prompters(state, [ref1, ref2])
    state["reference_boxes"] = reference_boxes
    state["confirmed"] = {}
    refs = set(state["reference_indices"])
    results = {}
    errors = []
    target_paths = [(i, img_path) for i, img_path in enumerate(state["image_paths"]) if i not in refs]
    max_target_frames = int(max_target_frames or 0)
    if max_target_frames > 0:
        target_paths = target_paths[:max_target_frames]
    for _, img_path in target_paths:
        frame_name = Path(img_path).name
        try:
            result = trex2.visual_prompt_inference(pil_from_path(img_path), prompts)[0]
            if not isinstance(result, dict) or "scores" not in result or "boxes" not in result:
                raise RuntimeError(
                    f"Unexpected detection payload type={type(result)} keys={list(result.keys()) if isinstance(result, dict) else 'n/a'}"
                )
            scores = np.array(result["scores"], dtype=float)
            boxes = np.array(result["boxes"], dtype=float)
            keep = scores >= float(visual_threshold)
            results[frame_name] = {
                "frame": frame_name,
                "image_path": img_path,
                "scores": scores[keep].tolist(),
                "boxes": boxes[keep].tolist(),
            }
        except Exception as e:
            errors.append(f"{frame_name}: {compact_error_text(e)}")
    state["detections"] = results
    frame_keys = [k for k, v in results.items() if len(v["boxes"]) > 0]
    summary = f"Detections complete. Frames with candidates: {len(frame_keys)} / {len(target_paths)}"
    if errors:
        summary += "\nErrors:\n" + "\n".join(errors[:8])
        if len(errors) > 8:
            summary += f"\n... {len(errors) - 8} more"
    first_key = frame_keys[0] if frame_keys else None
    first_choices = [str(i) for i in range(len(results[first_key]["boxes"]))] if first_key is not None else []
    preview = render_detection_preview(state, first_key, 0, show_all_candidates) if first_key is not None else None
    cand_info = candidate_info_text(state, first_key, 0) if first_key is not None else "No detections above threshold"
    return (
        state,
        summary,
        gr.update(choices=frame_keys, value=first_key),
        0,
        gr.update(choices=first_choices, value=("0" if first_choices else None)),
        preview,
        cand_info,
        confirmed_summary(state),
    )


def render_detection_preview(state, frame_key, candidate_index, show_all_candidates=False):
    if frame_key is None:
        return None
    record = state["detections"].get(str(frame_key))
    if record is None:
        return None
    image_path = record.get("image_path") or str(Path(state["dataset_dir"]) / "Data" / record["frame"])
    img = pil_from_path(image_path)
    boxes = record["boxes"]
    scores = record["scores"]
    selected_boxes = []
    selected_scores = []
    selected_labels = []
    for i, box in enumerate(boxes):
        if i == candidate_index:
            selected_boxes.append(box)
            selected_scores.append(scores[i])
            selected_labels.append(f"selected#{i}")
    if show_all_candidates:
        img = draw_boxes(img, boxes, scores, color=(0, 180, 255), labels=[str(i) for i in range(len(boxes))])
    img = draw_boxes(img, selected_boxes, selected_scores, color=(255, 0, 0), labels=selected_labels)
    return np.array(img)


def candidate_info_text(state, frame_key, candidate_index):
    if frame_key is None:
        return "No frame selected"
    record = state["detections"].get(str(frame_key))
    if record is None or not record["boxes"]:
        return "No candidates in this frame"
    lines = [f"Frame: {record['frame']}", f"Candidates: {len(record['boxes'])}", f"Selected index: {candidate_index}"]
    for i, (score, box) in enumerate(zip(record["scores"], record["boxes"])):
        marker = "*" if i == candidate_index else " "
        lines.append(f"{marker} candidate {i}: score={score:.3f} box={[round(v, 1) for v in box]}")
    confirmed = state["confirmed"].get(str(frame_key))
    if confirmed is not None:
        lines.append(f"Confirmed: candidate {confirmed['candidate_index']}")
    return "\n".join(lines)


def update_candidate_view(state, frame_key, candidate_index, show_all_candidates=False):
    record = state["detections"].get(str(frame_key)) if frame_key is not None else None
    if record is None or not record["boxes"]:
        return 0, gr.update(choices=[], value=None), None, "No candidates in this frame"
    candidate_index = int(max(0, min(candidate_index, len(record["boxes"]) - 1)))
    choices = [str(i) for i in range(len(record["boxes"]))]
    return (
        candidate_index,
        gr.update(choices=choices, value=str(candidate_index)),
        render_detection_preview(state, frame_key, candidate_index, show_all_candidates),
        candidate_info_text(state, frame_key, candidate_index),
    )


def select_candidate_index(state, frame_key, candidate_value, show_all_candidates=False):
    if candidate_value is None or candidate_value == "":
        return update_candidate_view(state, frame_key, 0, show_all_candidates)
    return update_candidate_view(state, frame_key, int(candidate_value), show_all_candidates)


def accept_candidate(state, frame_key, candidate_index):
    if frame_key is None:
        raise gr.Error("Select a frame first")
    record = state["detections"].get(str(frame_key))
    if record is None or not record["boxes"]:
        raise gr.Error("No candidates in this frame")
    candidate_index = int(max(0, min(candidate_index, len(record["boxes"]) - 1)))
    state["confirmed"][str(frame_key)] = {
        "frame": record["frame"],
        "candidate_index": candidate_index,
        "box": record["boxes"][candidate_index],
        "score": record["scores"][candidate_index],
        "source": "inference",
    }
    return state, candidate_info_text(state, frame_key, candidate_index), confirmed_summary(state)


def reject_frame(state, frame_key):
    if frame_key is not None:
        state["confirmed"].pop(str(frame_key), None)
    return state, confirmed_summary(state)


def clear_current_observations(state):
    if state is None:
        raise gr.Error("Load a dataset first")
    state["reference_boxes"] = []
    state["confirmed"] = {}
    state["detections"] = {}
    return state, "Cleared current object observations. Existing fitted OBB JSON entries were kept."


def confirmed_summary(state):
    lines = []
    for value in state.get("reference_boxes", []):
        lines.append(f"{value['frame']}: reference box")
    for _, value in sorted(state["confirmed"].items(), key=lambda kv: kv[1]["frame"]):
        lines.append(f"{value['frame']}: candidate {value['candidate_index']} score={value['score']:.3f}")
    return "\n".join(lines) if lines else "No confirmed detections yet"


def frame_index_from_name(name):
    m = re.search(r"(\d+)", name)
    return str(int(m.group(1))) if m else None


def pose_matrix_for_frame(state, frame_name):
    idx = frame_index_from_name(frame_name)
    return np.array(state["poses"][idx], dtype=np.float64)


def obb_corners(center, extent, rotation):
    center = np.asarray(center, dtype=np.float64)
    extent = np.asarray(extent, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    signs = np.array(
        [[sx, sy, sz] for sx in [-0.5, 0.5] for sy in [-0.5, 0.5] for sz in [-0.5, 0.5]],
        dtype=np.float64,
    )
    return center + (signs * extent) @ rotation.T


def project_points(K, points_world, T_world_to_cam):
    pts = np.asarray(points_world, dtype=np.float64)
    homog = np.c_[pts, np.ones(len(pts))]
    cam = (T_world_to_cam @ homog.T).T[:, :3]
    uvw = (K @ cam.T).T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-9)
    return uv, cam[:, 2]


def project_obb_box(state, center, extent, rotation, frame_name, use_inverse_pose=False):
    T = pose_matrix_for_frame(state, frame_name)
    if use_inverse_pose:
        T = np.linalg.inv(T)
    uv, z = project_points(
        np.array(state["intrinsic"]["camera_matrix"], dtype=np.float64),
        obb_corners(center, extent, rotation),
        T,
    )
    if np.any(z <= 0):
        return None
    return np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()], dtype=np.float64)


def camera_center_and_ray_world(state, frame_name, uv, use_inverse_pose=False):
    K = np.array(state["intrinsic"]["camera_matrix"], dtype=np.float64)
    T = pose_matrix_for_frame(state, frame_name)
    T_wc = np.linalg.inv(T) if not use_inverse_pose else T
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


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def collapse_observations_by_frame(observations):
    kept = {}
    dropped = []
    for obs in observations:
        frame = obs["frame"]
        prev = kept.get(frame)
        if prev is None:
            kept[frame] = obs
            continue
        if box_area(obs["box"]) > box_area(prev["box"]):
            dropped.append(prev)
            kept[frame] = obs
        else:
            dropped.append(obs)
    return [kept[k] for k in sorted(kept)], dropped


def observation_points(obs):
    x1, y1, x2, y2 = obs["box"]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return {
        "center": [cx, cy],
        "top_center": [cx, y1],
        "bottom_center": [cx, y2],
        "left_center": [x1, cy],
        "right_center": [x2, cy],
    }


def triangulate_anchor_set(state, observations, key, use_inverse_pose=True):
    ray_origins, ray_dirs = [], []
    for obs in observations:
        C, d = camera_center_and_ray_world(state, obs["frame"], observation_points(obs)[key], use_inverse_pose)
        ray_origins.append(C)
        ray_dirs.append(d)
    return triangulate_point_from_rays(ray_origins, ray_dirs)


def average_camera_center(state, frames):
    centers = []
    for frame_name in frames:
        T_wc = pose_matrix_for_frame(state, frame_name)
        centers.append(T_wc[:3, 3])
    return np.mean(np.asarray(centers, dtype=np.float64), axis=0)


def orthonormal_basis_from_axis(axis, center, state, frames):
    z_axis = np.asarray(axis, dtype=np.float64)
    z_axis = z_axis / max(np.linalg.norm(z_axis), 1e-9)
    facing = average_camera_center(state, frames) - center
    facing = facing - np.dot(facing, z_axis) * z_axis
    if np.linalg.norm(facing) < 1e-6:
        seed = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        facing = seed - np.dot(seed, z_axis) * z_axis
    x_axis = facing / np.linalg.norm(facing)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def make_planar_facing_rotation(center, state, frames):
    normal = average_camera_center(state, frames) - center
    normal = normal / max(np.linalg.norm(normal), 1e-9)
    x_axis = np.cross(WORLD_UP, normal)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    return np.column_stack([x_axis, y_axis, normal])


def build_geometric_initialization(state, observations, hyp, use_inverse_pose=True):
    frames = [obs["frame"] for obs in observations]
    center0 = triangulate_anchor_set(state, observations, "center", use_inverse_pose)
    bottom0 = triangulate_anchor_set(state, observations, "bottom_center", use_inverse_pose)
    top0 = triangulate_anchor_set(state, observations, "top_center", use_inverse_pose)
    axis_tb = top0 - bottom0
    if np.linalg.norm(axis_tb) < 1e-6:
        axis_tb = WORLD_UP.copy()
    axis_tb = axis_tb / np.linalg.norm(axis_tb)

    if hyp["mode"] == "center_compact":
        rotation = np.eye(3, dtype=np.float64)
        center_seed = center0
        support_z = bottom0[2]
        extent0 = np.array(hyp["extent0"], dtype=np.float64)
    elif hyp["mode"] == "top_bottom_axis":
        rotation = orthonormal_basis_from_axis(axis_tb, center0, state, frames)
        height0 = max(0.03, float(np.linalg.norm(top0 - bottom0)))
        extent0 = np.array([hyp["extent0"][0], hyp["extent0"][1], height0], dtype=np.float64)
        center_seed = 0.5 * (top0 + bottom0)
        support_z = bottom0[2]
    elif hyp["mode"] == "top_bottom_axis_support":
        rotation = orthonormal_basis_from_axis(axis_tb, center0, state, frames)
        height0 = max(0.03, float(np.linalg.norm(top0 - bottom0)))
        extent0 = np.array([hyp["extent0"][0], hyp["extent0"][1], height0], dtype=np.float64)
        bottom_snap = bottom0.copy()
        bottom_snap[2] = np.median([bottom0[2], center0[2] - 0.5 * height0, top0[2] - height0])
        center_seed = bottom_snap + 0.5 * height0 * rotation[:, 2]
        support_z = bottom_snap[2]
    elif hyp["mode"] == "upright_support":
        rotation = orthonormal_basis_from_axis(WORLD_UP, center0, state, frames)
        height0 = max(0.03, abs(top0[2] - bottom0[2]))
        extent0 = np.array([hyp["extent0"][0], hyp["extent0"][1], max(height0, hyp["extent0"][2])], dtype=np.float64)
        center_seed = bottom0 + 0.5 * extent0[2] * WORLD_UP
        support_z = bottom0[2]
    else:
        rotation = make_planar_facing_rotation(center0, state, frames)
        center_seed = center0
        support_z = bottom0[2]
        extent0 = np.array(hyp["extent0"], dtype=np.float64)
    return center_seed, extent0, rotation, float(support_z)


def observation_edge_errors(state, observations, center, extent, rotation, use_inverse_pose=True):
    errors = []
    for obs in observations:
        pred = project_obb_box(state, center, extent, rotation, obs["frame"], use_inverse_pose=use_inverse_pose)
        if pred is None:
            errors.append(1e6)
            continue
        target = np.array(obs["box"], dtype=np.float64)
        width = max(10.0, target[2] - target[0])
        height = max(10.0, target[3] - target[1])
        err = np.array(
            [
                (pred[0] - target[0]) / width,
                (pred[2] - target[2]) / width,
                (pred[1] - target[1]) / height,
                (pred[3] - target[3]) / height,
            ],
            dtype=np.float64,
        )
        errors.append(float(np.linalg.norm(err)))
    return errors


def fit_geometric_hypothesis(state, observations, hyp, use_inverse_pose=True):
    center_seed, extent_seed, rotation, support_z = build_geometric_initialization(
        state,
        observations,
        hyp,
        use_inverse_pose=use_inverse_pose,
    )
    prior_weight = float(hyp["prior_weight"])

    def residual(params, obs_set):
        center = params[:3]
        extent = np.exp(params[3:6])
        res = []
        for obs in obs_set:
            pred = project_obb_box(state, center, extent, rotation, obs["frame"], use_inverse_pose=use_inverse_pose)
            if pred is None:
                res.extend([1000, 1000, 1000, 1000])
                continue
            target = np.array(obs["box"], dtype=np.float64)
            width = max(10.0, target[2] - target[0])
            height = max(10.0, target[3] - target[1])
            res.extend(
                [
                    (pred[0] - target[0]) / width,
                    (pred[2] - target[2]) / width,
                    (pred[1] - target[1]) / height,
                    (pred[3] - target[3]) / height,
                ]
            )
        res.extend(((np.log(extent) - np.log(extent_seed)) * prior_weight).tolist())
        if hyp["mode"] in {"top_bottom_axis_support", "upright_support"}:
            bottom_center = center - 0.5 * extent[2] * rotation[:, 2]
            res.append((bottom_center[2] - support_z) * 10.0)
        return np.asarray(res, dtype=np.float64)

    x0 = np.r_[center_seed, np.log(extent_seed)]
    first = least_squares(lambda p: residual(p, observations), x0, loss="soft_l1", f_scale=1.0, max_nfev=2500)
    first_center = first.x[:3]
    first_extent = np.exp(first.x[3:6])
    per_obs = observation_edge_errors(state, observations, first_center, first_extent, rotation, use_inverse_pose)

    refined_obs = observations
    if len(observations) >= 5:
        order = np.argsort(per_obs)
        keep_n = max(3, int(math.ceil(0.8 * len(observations))))
        keep_idx = sorted(order[:keep_n].tolist())
        refined_obs = [observations[i] for i in keep_idx]

    second = least_squares(lambda p: residual(p, refined_obs), first.x, loss="soft_l1", f_scale=1.0, max_nfev=2500)
    center = second.x[:3]
    extent = np.exp(second.x[3:6])
    all_errors = observation_edge_errors(state, observations, center, extent, rotation, use_inverse_pose)
    return {
        "hypothesis": hyp["name"],
        "mode": hyp["mode"],
        "center": center.tolist(),
        "extent": extent.tolist(),
        "rotation": rotation.tolist(),
        "cost": float(second.cost),
        "mean_edge_error": float(np.mean(all_errors)),
        "kept_observations": len(refined_obs),
        "num_observations": len(observations),
        "per_observation_error": all_errors,
    }


def fit_auto_geometric_obb(state, entity_name, observations, use_inverse_pose=True):
    candidates = [fit_geometric_hypothesis(state, observations, hyp, use_inverse_pose) for hyp in GEOMETRIC_HYPOTHESES]
    candidates = sorted(candidates, key=lambda x: (x["mean_edge_error"], x["cost"]))
    best = candidates[0]
    return {
        "entity": entity_name,
        "obb": {
            "center": best["center"],
            "extent": best["extent"],
            "rotation": best["rotation"],
        },
        "fit": {
            "method": "auto_geometric",
            "hypothesis": best["hypothesis"],
            "mean_edge_error": best["mean_edge_error"],
            "cost": best["cost"],
            "kept_observations": best["kept_observations"],
            "num_observations": best["num_observations"],
            "all_hypotheses": candidates,
        },
    }


def upsert_obb_result(state, result):
    results = [row for row in state.get("obb_results", []) if row.get("entity") != result.get("entity")]
    results.append(result)
    state["obb_results"] = results
    state["obb_result"] = result


def fit_obb_from_confirmed(state, entity_name, mode, use_inverse_pose, extent_x, extent_y, extent_z):
    observations = list(state.get("reference_boxes", [])) + list(state["confirmed"].values())
    observations, dropped_duplicates = collapse_observations_by_frame(observations)
    unique_frames = sorted({obs["frame"] for obs in observations})
    if len(unique_frames) < 2:
        raise gr.Error("Need observations from at least two frames to fit a 3D box")

    if mode == "auto_geometric":
        obb = fit_auto_geometric_obb(state, entity_name, observations, use_inverse_pose=use_inverse_pose)
        obb["dropped_duplicate_observations"] = dropped_duplicates
        upsert_obb_result(state, obb)
        return state, json.dumps(obb, indent=2), render_obb_preview_html(state, use_inverse_pose), export_json_text(state, entity_name)

    ray_origins, ray_dirs = [], []
    for obs in observations:
        x1, y1, x2, y2 = obs["box"]
        uv = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
        C, d = camera_center_and_ray_world(state, obs["frame"], uv, use_inverse_pose=use_inverse_pose)
        ray_origins.append(C)
        ray_dirs.append(d)
    center0 = triangulate_point_from_rays(ray_origins, ray_dirs)

    if mode == "identity":
        R0 = np.eye(3, dtype=np.float64)
    else:
        raise gr.Error(f"Unsupported manual fit mode: {mode}")

    e0 = np.array([extent_x, extent_y, extent_z], dtype=np.float64)

    def residual(params):
        center = params[:3]
        extent = np.exp(params[3:6])
        res = []
        for obs in observations:
            pred = project_obb_box(state, center, extent, R0, obs["frame"], use_inverse_pose=use_inverse_pose)
            if pred is None:
                res.extend([1000, 1000, 1000, 1000])
                continue
            target = np.array(obs["box"], dtype=np.float64)
            scale = max(10.0, math.sqrt(max(1.0, (target[2] - target[0]) * (target[3] - target[1]))))
            res.extend(((pred - target) / scale).tolist())
        res.extend(((np.log(extent) - np.log(e0)) * 0.15).tolist())
        return np.array(res, dtype=np.float64)

    result = least_squares(residual, np.r_[center0, np.log(e0)], loss="soft_l1", f_scale=1.0, max_nfev=2000)
    obb = {
        "entity": entity_name,
        "obb": {
            "center": result.x[:3].tolist(),
            "extent": np.exp(result.x[3:6]).tolist(),
            "rotation": R0.tolist(),
        },
        "cost": float(result.cost),
        "dropped_duplicate_observations": dropped_duplicates,
    }
    upsert_obb_result(state, obb)
    return state, json.dumps(obb, indent=2), render_obb_preview_html(state, use_inverse_pose), export_json_text(state, entity_name)


def encode_preview_image(img):
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=88)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def render_obb_preview_html(state, use_inverse_pose):
    results = state.get("obb_results", [])
    if not results:
        return "<p>No OBBs fitted yet.</p>"
    frame_names = [Path(p).name for p in state.get("image_paths", [])]
    if not frame_names:
        frame_names = sorted(
            {v["frame"] for v in state.get("reference_boxes", [])}
            | {v["frame"] for v in state.get("confirmed", {}).values()}
        )
    if not frame_names:
        return "<p>No dataset frames available for preview.</p>"

    colors = [
        (255, 0, 0),
        (0, 128, 255),
        (0, 180, 0),
        (255, 140, 0),
        (180, 0, 180),
        (0, 170, 170),
        (220, 0, 120),
    ]
    edges = [(0, 1), (0, 2), (0, 4), (3, 1), (3, 2), (3, 7), (5, 1), (5, 4), (5, 7), (6, 2), (6, 4), (6, 7)]
    cards = []
    for frame_name in frame_names:
        img_path = Path(state["dataset_dir"]) / "Data" / frame_name
        if not img_path.exists():
            continue
        img = pil_from_path(img_path)
        T = pose_matrix_for_frame(state, frame_name)
        if use_inverse_pose:
            T = np.linalg.inv(T)
        draw = ImageDraw.Draw(img)
        for idx, result in enumerate(results):
            obb = result["obb"]
            color = colors[idx % len(colors)]
            corners = obb_corners(np.array(obb["center"]), np.array(obb["extent"]), np.array(obb["rotation"]))
            uv, z = project_points(np.array(state["intrinsic"]["camera_matrix"], dtype=np.float64), corners, T)
            if np.any(z <= 0):
                continue
            uv = uv.astype(int)
            for a, b in edges:
                draw.line([tuple(uv[a]), tuple(uv[b])], fill=color, width=3)
            min_xy = uv.min(axis=0)
            draw.text((int(min_xy[0]), int(min_xy[1])), result["entity"], fill=color)
        img.thumbnail((520, 360))
        encoded = encode_preview_image(img)
        cards.append(
            f'<div style="display:inline-block; margin-right:12px; vertical-align:top;">'
            f'<img src="data:image/jpeg;base64,{encoded}" style="height:320px; width:auto; border:1px solid #ddd;">'
            f'<div style="font:12px sans-serif; margin-top:4px;">{frame_name}</div>'
            f'</div>'
        )
    legend = " ".join(
        f'<span style="display:inline-block; margin-right:14px;"><span style="display:inline-block; width:12px; height:12px; background:rgb{colors[idx % len(colors)]}; margin-right:4px;"></span>{result["entity"]}</span>'
        for idx, result in enumerate(results)
    )
    return (
        f'<div style="font:13px sans-serif; margin-bottom:8px;">{legend}</div>'
        '<div style="overflow-x:auto; white-space:nowrap; padding:8px; border:1px solid #ddd; border-radius:8px;">'
        + "".join(cards)
        + "</div>"
    )


def export_json_text(state, entity_name):
    entries = [{"entity": "vga_socket", "obb": VGA_OBB}] if entity_name != "vga_socket" else []
    for result in state.get("obb_results", []):
        if result["entity"] == "vga_socket":
            entries = [row for row in entries if row["entity"] != "vga_socket"]
        entries.append({"entity": result["entity"], "obb": result["obb"]})
    return json.dumps(entries, indent=2)


with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), title="T-Rex2 OBB App") as demo:
    app_state = gr.State(None)
    candidate_index_state = gr.State(0)

    gr.Markdown("## 1. Dataset")
    with gr.Row():
        dataset_dir = gr.Textbox(label="Dataset directory", value=DEFAULT_DATASET_DIR)
        load_btn = gr.Button("Load Dataset")
    load_summary = gr.Textbox(label="Load summary", interactive=False)
    gallery = gr.Gallery(label="Dataset frames", columns=4, height=420, allow_preview=True)
    ref_summary = gr.Textbox(label="Selected reference frames", interactive=False)

    gr.Markdown("## 2. Reference Selection")
    gr.Markdown("Click exactly two frames in the gallery to use as references, then press Populate Prompters.")
    with gr.Row():
        selected_reference = gr.Dropdown(label="Selected reference to remove", choices=[], allow_custom_value=False)
        remove_ref_btn = gr.Button("Remove From Reference Selection")
        populate_btn = gr.Button("Populate Reference Prompters")
    with gr.Row():
        ref1 = ImagePrompter(label="Reference 1", scale=1)
        ref2 = ImagePrompter(label="Reference 2", scale=1)

    gr.Markdown("## 3. Detection")
    with gr.Row():
        visual_threshold = gr.Slider(label="Visual threshold", value=0.3, minimum=0.0, maximum=1.0, step=0.01)
        max_target_frames = gr.Number(label="Max target frames (0 = all)", value=5, precision=0)
        show_all_candidates = gr.Checkbox(label="Show all candidates in preview", value=False)
        run_btn = gr.Button("Run T-Rex2 on non-reference frames")
    det_summary = gr.Textbox(label="Detection summary", interactive=False)

    gr.Markdown("## 4. Confirmation")
    with gr.Row():
        frame_choice = gr.Dropdown(label="Frame with candidates", choices=[])
        candidate_choice = gr.Dropdown(label="Candidate #", choices=[], allow_custom_value=False)
    candidate_info = gr.Textbox(label="Candidate info", lines=8, interactive=False)
    candidate_preview = gr.Image(label="Candidate preview (selected only unless all-candidates is enabled)")
    with gr.Row():
        accept_btn = gr.Button("Accept Current Candidate")
        reject_btn = gr.Button("Reject Frame")
        clear_obs_btn = gr.Button("Clear Current Object Observations")
    confirmed_box = gr.Textbox(label="Confirmed detections", lines=10, interactive=False)

    gr.Markdown("## 5. 3D OBB Fit")
    with gr.Row():
        entity_name = gr.Textbox(label="Entity name", value="unknown_object")
        fit_mode = gr.Radio(
            label="Fit mode",
            choices=["auto_geometric", "identity"],
            value="auto_geometric",
        )
        use_inverse_pose = gr.Checkbox(label="Use inverse pose convention", value=True)
    with gr.Row():
        extent_x = gr.Number(label="Extent X (m)", value=0.03)
        extent_y = gr.Number(label="Extent Y (m)", value=0.03)
        extent_z = gr.Number(label="Extent Z (m)", value=0.03)
        fit_btn = gr.Button("Fit 3D OBB")
    obb_text = gr.Textbox(label="OBB result", lines=16)

    gr.Markdown("## 6. Visualization")
    obb_preview = gr.HTML(label="Projected OBB preview for all frames")

    gr.Markdown("## 7. Export")
    export_text = gr.Textbox(label="Teacher-format JSON", lines=20)

    load_btn.click(build_gallery, inputs=[dataset_dir], outputs=[app_state, gallery, load_summary, selected_reference])
    gallery.select(select_reference, inputs=[app_state], outputs=[app_state, ref_summary, selected_reference])
    remove_ref_btn.click(
        remove_selected_reference,
        inputs=[app_state, selected_reference],
        outputs=[app_state, ref_summary, selected_reference],
    )
    populate_btn.click(populate_reference_prompters, inputs=[app_state], outputs=[ref1, ref2])
    run_btn.click(
        run_trex_detection,
        inputs=[app_state, ref1, ref2, visual_threshold, max_target_frames, show_all_candidates],
        outputs=[app_state, det_summary, frame_choice, candidate_index_state, candidate_choice, candidate_preview, candidate_info, confirmed_box],
    )
    frame_choice.change(
        update_candidate_view,
        inputs=[app_state, frame_choice, candidate_index_state, show_all_candidates],
        outputs=[candidate_index_state, candidate_choice, candidate_preview, candidate_info],
    )
    candidate_choice.change(
        select_candidate_index,
        inputs=[app_state, frame_choice, candidate_choice, show_all_candidates],
        outputs=[candidate_index_state, candidate_choice, candidate_preview, candidate_info],
    )
    show_all_candidates.change(
        update_candidate_view,
        inputs=[app_state, frame_choice, candidate_index_state, show_all_candidates],
        outputs=[candidate_index_state, candidate_choice, candidate_preview, candidate_info],
    )
    accept_btn.click(accept_candidate, inputs=[app_state, frame_choice, candidate_index_state], outputs=[app_state, candidate_info, confirmed_box])
    reject_btn.click(reject_frame, inputs=[app_state, frame_choice], outputs=[app_state, confirmed_box])
    clear_obs_btn.click(clear_current_observations, inputs=[app_state], outputs=[app_state, confirmed_box])
    fit_btn.click(
        fit_obb_from_confirmed,
        inputs=[app_state, entity_name, fit_mode, use_inverse_pose, extent_x, extent_y, extent_z],
        outputs=[app_state, obb_text, obb_preview, export_text],
    )


if __name__ == "__main__":
    args = arg_parse()
    trex2 = TRex2APIWrapper(args.trex2_api_token)
    demo.launch(server_name=args.server_name, server_port=args.server_port)
