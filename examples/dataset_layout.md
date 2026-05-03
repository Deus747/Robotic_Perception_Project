# Expected Dataset Layout

The scripts assume the project dataset is available as:

```text
<dataset_root>/
  Data/
    frame_000319.png
    frame_000333.png
    ...
  intrinsic.json
  poses.json
  sample_answers.json
```

YOLO training additionally needs a standard YOLO dataset YAML:

```yaml
path: /absolute/path/to/yolo_dataset
train: images/train
val: images/val
names:
  0: power_socket
  1: ethernet_port
  2: vga_port
```

