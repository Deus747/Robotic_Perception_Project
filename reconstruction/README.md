# 3D OBB Reconstruction

This folder contains the reconstruction-stage scripts copied from the active project branch.

## Recommended Script

Use:

```powershell
python reconstruction/refit_nonml_geometric_obbs.py
```

This is the current recommended generic fitter. It:

- uses inverse pose convention,
- collapses duplicate detections of the same class in the same frame to the largest box,
- tests center-compact, top/bottom-axis, support-contact, upright-support, and planar-facing hypotheses,
- writes projected overlays and `obb_estimates.json`.

The T-Rex2 app's `auto_geometric` mode was parity-checked against this script and matches within numerical tolerance.

## Expected Input

The original scripts expect the project-style layout:

```text
Data/
intrinsic.json
poses.json
obb_reconstruction_inputs/yolo26_9class_sahi_rerun_no_purifier/detections_flat.json
```

Use `supervised/sahi_yolo_inference.py` or the app/export workflow to produce detections in the same flat JSON format.

