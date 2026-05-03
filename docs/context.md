# Robotic Perception Project Context

## Scope

This file is the current continuation context for the project. It is focused on:

- known-object 2D detection using the trained YOLO 9-class model,
- SAHI-based inference on the released frames,
- reconstruction-oriented export of 2D detections,
- multiple 2D -> 3D OBB fitting attempts,
- current conclusions about what is and is not working.

This version intentionally de-emphasizes the older T-Rex app branch. The active work is now the YOLO26 + SAHI + 3D reconstruction branch.

## Project framing

The project is about metric-semantic reconstruction from posed RGB images.

For the OBB component, the expected output is a strict JSON file containing 3D oriented bounding boxes for requested scene entities. The teacher provided one sample OBB for:

- `vga_socket`

The baseline requested objects are:

- `ethernet_socket`
- `power_socket`

There may also be one or two additional objects requested on the final day.

The evaluator projects submitted 3D OBBs into hidden test images from the same scene/capture and computes polygon IoU against ground truth. Hidden test images are missing frames from the same scene, so the final OBB JSON is the actual deliverable for that part.

## Dataset

Main files in the workspace:

- `Data/*.png`
- `intrinsic.json`
- `poses.json`
- `sample_answers.json`
- `Final_Project.pdf`

Important note:

- `sample_answers.json` is only a template. It contains placeholder tokens and is not valid final JSON.

## Core assumption currently in use

The correct pose convention for this dataset appears to require **inverse pose usage** for projection/backprojection. All recent reconstruction overlays and the VGA sample comparison were done with that convention because it was the only one producing sensible image projections.

## Detection work completed

### Known-object detector

A trained YOLO 9-class detector is available at:

- `yolo26_9class_results/base_weights_9class/weights/best.pt`

Classes present in the current 9-class setup include:

- `power_socket`
- `ethernet_port`
- `vga_port`
- `bottle`
- `key`
- `glass`
- `purifier`
- `ps`
- `sticker`

### Initial normalized export from existing SAHI detections

Script:

- `prepare_yolo26_9class_obb_inputs.py`

This normalized an already existing SAHI run into a reconstruction-ready format.

Output folder:

- `obb_reconstruction_inputs/yolo26_9class_sahi/`

Files:

- `detections_by_frame.json`
- `detections_flat.json`
- `metadata.json`
- `visualizations/`

This export was based on existing detections from:

- `yolo26_9class_results/sahi_results_9class/detections.json`

Summary from that export:

- 16 frames
- 117 detections total

Class counts there were:

- `power_socket`: 14
- `ethernet_port`: 14
- `vga_port`: 15
- `bottle`: 10
- `key`: 5
- `glass`: 11
- `purifier`: 20
- `ps`: 14
- `sticker`: 14

### Fresh SAHI rerun with current weight, purifier skipped

Script:

- `rerun_yolo26_9class_sahi_and_reconstruct.py`

Purpose:

1. rerun SAHI using the current 9-class weight,
2. skip `purifier` because it is large and SAHI was creating multiple instances,
3. export reconstruction-ready 2D detections,
4. run a simple first-pass 3D OBB reconstruction,
5. visualize projected OBBs back on the images.

Primary output folder:

- `obb_reconstruction_inputs/yolo26_9class_sahi_rerun_no_purifier/`

Key files:

- `detections_by_frame.json`
- `detections_flat.json`
- `obb_estimates.json`
- `metadata.json`
- `reconstruction_3d.png`
- `visualizations_2d/`
- `visualizations_3d_overlays/`

The current rerun configuration was executed at a confidence threshold of about `0.50`.

Current metadata after the rerun:

- 16 frames
- 96 detections
- `purifier` skipped

Current class counts:

- `bottle`: 10
- `ethernet_port`: 14
- `glass`: 11
- `key`: 5
- `power_socket`: 14
- `ps`: 14
- `sticker`: 14
- `vga_port`: 14

## Current reconstruction scripts

### 1. First-pass reconstruction from 2D detections

Script:

- `rerun_yolo26_9class_sahi_and_reconstruct.py`

Behavior:

- uses the rerun SAHI detections,
- triangulates object position from multi-view observations,
- applies a lightweight prior-based OBB fit,
- renders a 3D plot of the reconstructed scene,
- projects each OBB back into the original images.

This script was updated once to strengthen some priors:

- bottle / glass: more upright prior
- sticker / key: more planar-facing prior
- panel objects: stronger regularization

Even after that update, the bottle remained physically poor.

### 2. Generic multi-hypothesis refit

Script:

- `refit_generic_obbs.py`

Purpose:

- move away from class-specific reconstruction assumptions,
- try a more generic 2D -> 3D fitting strategy suitable for unknown final-day objects.

It loads detections from:

- `obb_reconstruction_inputs/yolo26_9class_sahi_rerun_no_purifier/detections_flat.json`

It tries combinations of:

- anchor modes:
  - `center`
  - `bottom_center`
- rotation modes:
  - `identity`
  - `camera_facing`
  - `upright`
  - `planar_facing`
- generic shape priors:
  - compact
  - upright
  - flat
  - elongated

It also rejects the worst 20% of observations when enough views are available.

Output folder:

- `obb_reconstruction_inputs/yolo26_9class_generic_refit/`

Key files:

- `obb_estimates.json`
- `all_hypotheses.json`
- `metadata.json`
- `reconstruction_3d.png`
- `visualizations_3d_overlays/`

Observed chosen hypotheses from the successful run:

- `bottle`: `compact_center_identity`
- `ethernet_port`: `compact_center_identity`
- `glass`: `upright_center`
- `key`: `compact_center_identity`
- `power_socket`: `compact_center_identity`
- `ps`: `elongated_center`
- `sticker`: `compact_center_identity`
- `vga_port`: `elongated_center`

Conclusion from this stage:

- generic multi-hypothesis fitting helps some classes,
- but 2D bounding boxes alone still underconstrain objects like the bottle,
- the optimizer still prefers a compact physically wrong explanation if it lowers box reprojection error.

### 3. Mask-based generic refit

Script:

- `refit_mask_obbs.py`

Purpose:

- test whether weak masks extracted from the 2D detections improve the generic reconstruction.

Method:

- per-detection GrabCut mask generation,
- mask centroid and mask bottom-center anchor extraction,
- mask hull computation,
- combined ranking using:
  - projected OBB box error,
  - projected OBB polygon overlap with the mask hull.

Output folder:

- `obb_reconstruction_inputs/yolo26_9class_mask_refit/`

Key files:

- `obb_estimates.json`
- `all_hypotheses.json`
- `mask_observations.json`
- `metadata.json`
- `reconstruction_3d.png`
- `visualizations_3d_overlays/`
- `masks/`

Observed chosen hypotheses from the successful run:

- `bottle`: `compact_mask_centroid`
- `ethernet_port`: `compact_mask_centroid`
- `glass`: `camera_mask_centroid`
- `key`: `compact_mask_centroid`
- `power_socket`: `upright_centroid`
- `ps`: `compact_mask_centroid`
- `sticker`: `compact_bottom`
- `vga_port`: `compact_bottom`

Conclusion from this stage:

- masks helped some classes such as `sticker`, `vga_port`, `ethernet_port`, `glass`, `ps`,
- bottle was still bad,
- `power_socket` actually degraded under this mask-based generic refit,
- GrabCut masks are not reliable enough for tiny dark connectors,
- weak masks alone are not enough to solve free-standing upright objects.

## VGA sample comparison

The current fitted `vga_port` OBB was compared against the provided sample `vga_socket` OBB by:

1. projecting both 3D OBBs into all 16 released frames,
2. taking the projected convex polygon of each OBB,
3. computing polygon IoU per frame,
4. averaging across frames.

Result:

- mean polygon IoU: `0.7129`
- min IoU: `0.6691`
- max IoU: `0.7827`

Interpretation:

- the current VGA rotation matches the sample,
- the current VGA center is very close to the sample,
- the main difference is extent inflation,
- so the current VGA is geometrically close but not yet tight enough.

This was useful as a sanity check for the 2D -> 3D pipeline and for validating the inverse-pose convention.

## Current qualitative failure modes

### Bottle

This was the main generic failure case before the duplicate-observation fix.

Root cause found:

- `frame_000461.png` contained two `bottle` detections:
  - one normal full-bottle detection,
  - one tiny bottom-fragment false duplicate with high confidence.
- the reconstruction code grouped all same-class detections as one object,
- this meant the tiny duplicate was treated as another independent view of the bottle,
- the corrupted observation set caused the optimizer to prefer a compact/flat physically wrong solution.

Fix applied:

- reconstruction scripts now collapse duplicate detections of the same class in the same frame to the largest box before 3D fitting,
- raw detection exports are left intact,
- metadata records dropped duplicate observations.

Current result after rerunning `refit_nonml_geometric_obbs.py`:

- bottle hypothesis changed from `center_compact` to `tb_axis`,
- bottle mean edge error improved from about `1.0237` to `0.0987`,
- bottle OBB is now tall and much more plausible:
  - center: `[-0.5357768739967653, 2.615193255826356, 0.6904258828156151]`
  - extent: `[0.06047575095325525, 0.12238136066952343, 0.3347162624100818]`

Previous observed issues:

- wrong global position offset,
- wrong physical scale,
- sometimes appears effectively like a small misplaced 3D object,
- generic fitting prefers a compact solution that explains 2D rectangles but is not physically plausible.

What this means:

- for the released bottle detections, the dominant issue was an implementation/association bug rather than a pure geometric limitation,
- support-plane/stronger segmentation may still help unknown free-standing objects, but the bottle should no longer be used as evidence that the basic fitter is broken.

### Sticker and power_socket

These are better than bottle but still show some visible orientation or extent issues in the projected overlays.

### VGA / ethernet / some others

These are comparatively more stable under the current pipeline.

## Current technical conclusion

The main bottleneck is no longer 2D detection quality alone. The bottleneck is the **generic 2D -> 3D lift**.

What has been established:

- the trained YOLO 9-class model plus SAHI can produce usable multi-view 2D detections,
- inverse pose convention is the correct projection convention for this dataset,
- generic multi-hypothesis fitting is better than a single naive fit,
- weak mask extraction helps some objects,
- but **2D boxes and weak masks alone are insufficient for physically correct generic 3D OBB recovery**, especially for objects like the bottle.

## What is likely needed next

The strongest next generic improvement is likely one of:

1. support-plane estimation,
2. stronger segmentation than GrabCut,
3. real depth / point cloud usage if available,
4. a better reconstruction loss that uses more than 2D rectangular fit.

If the goal is to remain generic for final-day unknown objects, support-plane estimation is the most promising next direction because it helps free-standing objects without hardcoding object categories.

## Other files added in this branch

- `project_report_detection.tex`
  - Overleaf-ready LaTeX report focused on the detection pipeline and unknown-object detection progression.

## Bottom line

The current active branch is:

```text
trained YOLO 9-class model
-> SAHI inference on released frames
-> reconstruction-ready 2D detection export
-> generic 2D -> 3D OBB fitting experiments
-> overlay-based qualitative validation
```

Current best evidence:

- detection is usable,
- reconstruction is partially working,
- the main unresolved problem is robust generic 2D -> 3D OBB recovery for objects whose 2D boxes do not strongly constrain physical 3D size and pose.
