#!/usr/bin/env python3

import argparse
import json
import shutil
import os
from pathlib import Path
from typing import Dict, List, Optional
import yaml

def convert_bbox_coco_to_yolo(bbox: List[float], img_width: int, img_height: int) -> List[float]:
    """
    Convert COCO bbox [x_min, y_min, width, height] to YOLO format 
    [x_center, y_center, width, height] (normalized)
    """
    x_min, y_min, width, height = bbox
    
    # Convert to center coordinates
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    
    # Normalize coordinates
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    return [x_center, y_center, width, height]

def process_split(
    coco_file: Path,
    output_dir: Path,
    split_name: str,
    source_dir: Path,
    include_empty: bool = True
) -> Optional[Dict]:
    """
    Process a single data split (train/val/test)
    """
    if not coco_file.exists():
        print(f"Skipping {split_name} split - file not found: {coco_file}")
        return None
    
    print(f"\nProcessing {split_name} split...")
    
    # Load COCO annotations
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directories
    images_dir = output_dir / 'images' / split_name
    labels_dir = output_dir / 'labels' / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Track statistics
    total_images = len(coco_data['images'])
    processed_images = 0
    skipped_images = 0
    
    # Create image id to info mapping
    image_info = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image
    image_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)
    
    # Process each image
    for img_id, img_data in image_info.items():
        # Copy image file
        img_filename = img_data['file_name']
        source_path = source_dir / img_filename
        if not source_path.exists():
            print(f"Warning: Source image not found: {source_path}")
            skipped_images += 1
            continue
            
        dest_path = images_dir / source_path.name
        shutil.copy2(source_path, dest_path)
        
        # Create label file
        label_path = labels_dir / f"{source_path.stem}.txt"
        
        # Convert annotations for this image
        if img_id in image_annotations:
            with open(label_path, 'w') as f:
                for ann in image_annotations[img_id]:
                    bbox = convert_bbox_coco_to_yolo(
                        ann['bbox'],
                        img_data['width'],
                        img_data['height']
                    )
                    # Convert 1-based COCO category ID to 0-based YOLO class ID
                    class_id = ann['category_id'] - 1
                    f.write(f"{class_id} {' '.join(map(str, bbox))}\n")
        elif include_empty:
            # Create empty label file for images with no annotations
            label_path.touch()
        
        processed_images += 1
    
    print(f"Split: {split_name}")
    print(f"  Total images in annotations: {total_images}")
    print(f"  Successfully processed: {processed_images}")
    if skipped_images > 0:
        print(f"  Skipped (images not found): {skipped_images}")
    
    return coco_data['categories']

def create_yaml_file(output_dir: Path, categories: List[Dict], splits: List[str]):
    """Create data.yaml file for ultralytics YOLOv8"""
    # Convert 1-based COCO category IDs to 0-based YOLO class indices
    names = [cat['name'] for cat in sorted(categories, key=lambda x: x['id'])]
    
    yaml_data = {
        'path': str(output_dir.absolute()),
        'train': 'train',
        'val': 'val',
        'names': names  # YOLO expects a list of names, indexed from 0
    }
    
    if 'test' in splits:
        yaml_data['test'] = 'test'
    
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)
    
    print(f"\nCreated data.yaml at {yaml_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert COCO format to YOLO format')
    parser.add_argument('input_dir', type=Path, help='Input directory containing COCO json files')
    parser.add_argument('output_dir', type=Path, help='Output directory for YOLO format dataset')
    parser.add_argument('--include-empty', action='store_true', default=False,
                      help='Include images with no annotations by creating empty label files (default: False)')
    parser.add_argument('--include-data-yaml', action='store_true', default=False,
                      help='Generate data.yaml file for YOLOv8 training (default: False)')
    parser.add_argument('--images-dir', type=Path,
                      help='Base directory for images. Will be prepended to image paths from COCO json')
    
    args = parser.parse_args()
    
    # If images_dir is not specified, use input_dir
    image_base_dir = args.images_dir if args.images_dir else args.input_dir
    
    print(f"\nInput directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Images directory: {image_base_dir}")
    
    # Verify input files exist
    train_file = args.input_dir / 'train.json'
    val_file = args.input_dir / 'val.json'
    test_file = args.input_dir / 'test.json'
    
    found_splits = []
    if train_file.exists():
        found_splits.append(('train', train_file))
    if val_file.exists():
        found_splits.append(('val', val_file))
    if test_file.exists():
        found_splits.append(('test', test_file))
    
    if not found_splits:
        print("Error: No split files (train.json, val.json, test.json) found in input directory")
        return
    
    print(f"\nFound split files: {[split[0] for split in found_splits]}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    splits = []
    categories = None
    
    for split_name, split_file in found_splits:
        splits.append(split_name)
        split_categories = process_split(
            split_file, 
            args.output_dir, 
            split_name, 
            image_base_dir,
            include_empty=args.include_empty
        )
        if not categories and split_categories:
            categories = split_categories
    
    if args.include_data_yaml and categories:
        create_yaml_file(args.output_dir, categories, splits)
    
    print("\nConversion complete!")
    print(f"Dataset created at: {args.output_dir}")

if __name__ == '__main__':
    main()
