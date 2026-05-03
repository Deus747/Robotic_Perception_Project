# T-Rex2 OBB App

Standalone Gradio app for:

1. loading a posed-image dataset
2. selecting two reference frames from the dataset gallery
3. drawing reference boxes
4. calling the T-Rex2 API on selected target frames
5. confirming detections per frame
6. fitting a 3D OBB from reference + confirmed boxes
7. exporting teacher-format JSON

This app is separate from the upstream `T-Rex` repository so it is easier to put in your own GitHub repo.

## What you need

- Python 3.10 or 3.11
- a T-Rex2 API token
- a dataset folder containing:
  - `Data/*.png`
  - `intrinsic.json`
  - `poses.json`

## Setup

Create a virtual environment and install dependencies:

```powershell
cd trex2_obb_app
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you already have the `T-Rex\.venv` environment working, you can reuse it. The launcher will automatically use:

1. `trex2_obb_app\.venv`
2. `..\T-Rex\.venv`
3. `python` from PATH

## Run

```powershell
cd trex2_obb_app
.\launch.ps1 -Token "YOUR_TREX2_API_TOKEN"
```

Optional:

```powershell
.\launch.ps1 -Token "YOUR_TREX2_API_TOKEN" -Port 7863
```

## Notes

- `Max target frames` exists so you can test the app without spending tokens on the whole dataset.
- Reference boxes are included in the 3D OBB fit.
- Candidate confirmation is done with a frame dropdown and a candidate-number dropdown.
- Candidate preview defaults to showing only the selected candidate for cleaner manual checking. Enable `Show all candidates in preview` when you need to compare all detections in a frame.
- Use `Remove From Reference Selection` to remove a frame from the selected-reference list without deleting the dataset image. Press `Populate Reference Prompters` again when you want the two reference boxes refreshed from the current selection.
- Fitted OBBs are cumulative. Fitting a new entity appends or replaces that entity in the teacher-format JSON and in the all-frame projected OBB visualization.
- Use `Clear Current Object Observations` before starting manual confirmation for the next object if you are not launching a fresh detection run.
- The projected OBB preview renders all dataset frames in a horizontal scroll region and overlays every fitted object currently stored in the app state.
- The default 3D fit mode is `auto_geometric`, which runs the stronger multi-hypothesis OBB fitter used in the main project:
  - center-compact,
  - top/bottom-axis,
  - top/bottom-axis with support contact,
  - upright support,
  - planar-facing.
- Duplicate observations in the same frame are collapsed to the largest box before fitting, which prevents small T-Rex/SAHI fragments from corrupting the 3D box.
- The app now defaults to the inverse pose convention, matching the projection checks used by the active reconstruction branch.
- OBB fit modes are intentionally limited to `auto_geometric` and `identity`; the old VGA-panel and camera-facing manual modes were removed from the UI to avoid using weaker reconstruction paths.
- This app uses the T-Rex2 HTTP API directly. It does not need a local T-Rex2 model checkout.
- The dataset directory textbox defaults to the parent project folder. Change it in the UI if your dataset is elsewhere.
