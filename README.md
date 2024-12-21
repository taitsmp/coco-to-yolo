# COCO YOLO Tools

Tools for working with COCO and YOLO format datasets.

## Installation

```bash
pip install git+https://github.com/taitsmp/coco-yolo-tools.git
```

## Usage

### Object Detection (Default)

```bash
coco-to-yolo /path/to/coco/dataset /path/to/output/yolo/dataset
```

Creates a YOLO format dataset with the following structure:
```
dataset/
  ├── images/
  │   ├── train/
  │   ├── val/
  │   └── test/
  └── labels/
      ├── train/
      ├── val/
      └── test/
```

### Classification

For classification datasets (where annotations only contain class labels, no bounding boxes):

```bash
coco-to-yolo /path/to/coco/dataset /path/to/output/yolo/dataset --classification
```

This will create a directory structure suitable for YOLOv8 classification:
```
dataset/
  ├── train/
  │   ├── class1/
  │   │   ├── image1.jpg
  │   │   ├── image2.jpg
  │   │   └── ...
  │   ├── class2/
  │   │   ├── image3.jpg
  │   │   ├── image4.jpg
  │   │   └── ...
  │   └── ...
  ├── val/
  │   ├── class1/
  │   │   └── ...
  │   ├── class2/
  │   │   └── ...
  │   └── ...
  └── test/
      ├── class1/
      │   └── ...
      ├── class2/
      │   └── ...
      └── ...
```

### Additional Options

- `--include-data-yaml`: Generate data.yaml file for YOLOv8 object detection training (not used for classification)
- `--include-empty`: (Object Detection only) Include images with no annotations
- `--images-dir`: Base directory for images if different from the input directory
