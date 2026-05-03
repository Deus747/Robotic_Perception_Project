# Supervised Detection Track

This folder recreates the known-object pipeline from the project notebooks without shipping the notebooks.

## Scripts

- `prepare_yolo_dataset_from_annotations.py`: converts simple bbox annotations into YOLO image/label folders and `data.yaml`.
- `train_yolo.py`: trains the YOLO known-object detector using the settings from the project report.
- `sahi_yolo_inference.py`: runs SAHI sliced inference and exports `detections_flat.json` / `detections_by_frame.json`.

## Recommended Known-Object Workflow

```powershell
python supervised/prepare_yolo_dataset_from_annotations.py `
  --annotations labels/annotations.json `
  --images Data `
  --out yolo_dataset `
  --class-map '{ "power_socket": 0, "ethernet_port": 1, "vga_port": 2 }'

python supervised/train_yolo.py `
  --data-yaml yolo_dataset/data.yaml `
  --weights yolo26s.pt `
  --project runs/yolo_known

python supervised/sahi_yolo_inference.py `
  --weights runs/yolo_known/train/weights/best.pt `
  --images Data `
  --out outputs/yolo_sahi
```

The core settings mirror the report: `imgsz=1024`, `epochs=100`, `batch=16`, `lr0=0.001`, `warmup_epochs=5`, and SAHI `512x512` slices with `0.30` overlap.

