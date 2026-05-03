# Notebook to Script Mapping

Full notebooks were intentionally not copied into this repository. The important reproducible sections were distilled as follows:

| Original notebook | Reproducible script |
| --- | --- |
| `RP_3class.ipynb` | `supervised/train_yolo.py`, `supervised/sahi_yolo_inference.py` |
| `RP_11class_merged.ipynb` | `supervised/train_yolo.py` with a larger `data.yaml` class list |
| `sahi_groundingdino_pipeline_colab.ipynb` | `unsupervised/groundingdino_sliced_inference.py` |
| `DINO_V2_SW.ipynb` | `unsupervised/dinov2_sliding_window.py` |
| `yolo_world_exemplar_pipeline_colab.ipynb` | `unsupervised/yoloworld_dinov2_exemplar.py` |
| T-Rex2 visual prompting branch | `app/` |

The report in `docs/project_report_detection.tex` explains why each method was tested and why later stages replaced earlier ones.

