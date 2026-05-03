# Unsupervised and Exemplar Detection Track

This folder contains lightweight Python versions of the important unknown-object notebook methods described in `docs/project_report_detection.tex`.

## Methods

- `groundingdino_sliced_inference.py`: open-vocabulary text prompting with manual SAHI-style tiling.
- `dinov2_sliding_window.py`: pure visual exemplar matching with multi-scale sliding windows.
- `yoloworld_dinov2_exemplar.py`: YOLO-World high-recall proposals reranked by DINOv2 similarity.
- `dinov2_utils.py` and `common.py`: shared embedding, NMS, visualization, and JSON export helpers.

## Intended Use

These scripts are not meant to outperform the T-Rex2 app. They document and reproduce the progression of unsupervised/exemplar experiments:

```text
text-only world model
-> sliced world model
-> DINOv2 visual search
-> YOLO-World proposal + DINOv2 reranking
-> T-Rex2 visual prompting app
```

For final manual unknown-object detection, use `app/`. For method comparison or report reproduction, use these scripts.

