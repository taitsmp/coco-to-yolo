#!/usr/bin/env python3

import argparse
import json
import shutil
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
    dir_name: str,
    category_id_to_index: Dict[int, int],
    include_empty: bool = True
) -> Optional[List[Dict]]:
    """
    Process a single data split (train/val/test)
    Uses a pre-computed global category_id_to_index mapping
    """
    if not coco_file.exists():
        print(f"Skipping {split_name} split - file not found: {coco_file}")
        return None
    
    print(f"\nProcessing {split_name} split...")
    
    # Load COCO annotations
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directories using custom directory name
    images_dir = output_dir / 'images' / dir_name
    labels_dir = output_dir / 'labels' / dir_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Track statistics
    total_images = len(coco_data['images'])
    processed_images = 0
    skipped_images = 0
    negative_examples = 0  # New counter for negative examples
    
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
            has_valid_annotations = False  # Track if any valid annotations exist
            with open(label_path, 'w') as f:
                for ann in image_annotations[img_id]:
                    # Check if bbox exists in annotation
                    if 'bbox' not in ann:
                        continue  # Skip annotations without bbox
                    bbox = convert_bbox_coco_to_yolo(
                        ann['bbox'],
                        img_data['width'],
                        img_data['height']
                    )
                    # Convert category ID to zero-based index using the mapping
                    class_id = category_id_to_index[ann['category_id']]
                    f.write(f"{class_id} {' '.join(map(str, bbox))}\n")
                    has_valid_annotations = True
            
            if not has_valid_annotations:
                negative_examples += 1
        elif include_empty:
            # Create empty label file for images with no annotations
            label_path.touch()
            negative_examples += 1
        
        processed_images += 1
    
    print(f"Split: {split_name}")
    print(f"  Total images in annotations: {total_images}")
    print(f"  Successfully processed: {processed_images}")
    if skipped_images > 0:
        print(f"  Skipped (images not found): {skipped_images}")
    if include_empty:
        print(f"  Negative examples (empty labels): {negative_examples}")
    
    # Print category ID mapping for user reference
    print("\nCategory ID mapping (original -> YOLO):")
    for cat in coco_data['categories']:
        print(f"  {cat['name']}: {cat['id']} -> {category_id_to_index[cat['id']]}")
    
    return coco_data['categories']

def create_yaml_file(output_dir: Path, categories: List[Dict], splits: List[str], train_dir: str, val_dir: str, test_dir: str):
    """Create data.yaml file for ultralytics YOLOv8"""
    # Convert 1-based COCO category IDs to 0-based YOLO class indices
    names = [cat['name'] for cat in sorted(categories, key=lambda x: x['id'])]
    
    yaml_data = {
        'path': '.',  # Use relative path instead of absolute
        'train': f'images/{train_dir}',  # Use custom directory names
        'val': f'images/{val_dir}',
        'names': names  # YOLO expects a list of names, indexed from 0
    }
    
    if 'test' in splits:
        yaml_data['test'] = f'images/{test_dir}'
    
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)
    
    print(f"\nCreated data.yaml at {yaml_path}")

def process_classification_split(
    coco_file: Path,
    output_dir: Path,
    split_name: str,
    source_dir: Path,
    dir_name: str,
) -> Optional[Dict]:
    """
    Process a single data split for classification data (train/val/test)
    """
    if not coco_file.exists():
        print(f"Skipping {split_name} split - file not found: {coco_file}")
        return None
    
    print(f"\nProcessing {split_name} split for classification...")
    
    # Load COCO annotations
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directory for this split using custom directory name
    split_dir = output_dir / dir_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    # Track statistics
    total_images = len(coco_data['images'])
    processed_images = 0
    skipped_images = 0
    
    # Create image id to info mapping
    image_info = {img['id']: img for img in coco_data['images']}
    
    # Create category id to name mapping
    category_map = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Group annotations by image
    image_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)
    
    # Process each image
    for img_id, img_data in image_info.items():
        # Get image annotations
        if img_id not in image_annotations:
            print(f"Warning: Image {img_id} has no annotations")
            skipped_images += 1
            continue
        
        # Get the category for this image (assuming single class per image)
        ann = image_annotations[img_id][0]
        category_id = ann['category_id']
        category_name = category_map[category_id]
        
        # Create category directory if it doesn't exist
        category_dir = split_dir / category_name
        category_dir.mkdir(exist_ok=True)
        
        # Copy image file to category directory
        img_filename = img_data['file_name']
        source_path = source_dir / img_filename
        if not source_path.exists():
            print(f"Warning: Source image not found: {source_path}")
            skipped_images += 1
            continue
            
        dest_path = category_dir / source_path.name
        shutil.copy2(source_path, dest_path)
        processed_images += 1
    
    print(f"Split: {split_name}")
    print(f"  Total images in annotations: {total_images}")
    print(f"  Successfully processed: {processed_images}")
    if skipped_images > 0:
        print(f"  Skipped (images not found or no annotations): {skipped_images}")
    
    return coco_data['categories']

def create_global_category_mapping(found_splits: List[Tuple[str, Path]]) -> Tuple[List[Dict], Dict[int, int]]:
    """Create a global mapping of category IDs to YOLO indices across all splits"""
    # Collect all categories from all splits
    all_categories = []
    for _, split_file in found_splits:
        if not split_file.exists():
            continue
        with open(split_file, 'r') as f:
            coco_data = json.load(f)
            all_categories.extend(coco_data['categories'])
    
    # Create name-based mapping to handle duplicates
    name_to_index = {}
    current_index = 0
    for cat in all_categories:
        if cat['name'] not in name_to_index:
            name_to_index[cat['name']] = current_index
            current_index += 1
    
    # Create mapping from original IDs to YOLO indices
    category_id_to_index = {cat['id']: name_to_index[cat['name']] for cat in all_categories}
    
    # Get unique categories (by name) for YAML file
    unique_categories = []
    seen_names = set()
    for cat in sorted(all_categories, key=lambda x: category_id_to_index[x['id']]):
        if cat['name'] not in seen_names:
            unique_categories.append(cat)
            seen_names.add(cat['name'])
    
    return unique_categories, category_id_to_index

def main():
    parser = argparse.ArgumentParser(description='Convert COCO format to YOLO format')
    parser.add_argument('input_dir', type=Path, help='Input directory containing COCO json files')
    parser.add_argument('output_dir', type=Path, help='Output directory for YOLO format dataset')
    parser.add_argument('--include-empty', action='store_true', default=False,
                      help='Include images with no annotations by creating empty label files (default: False)')
    parser.add_argument('--include-data-yaml', action='store_true', default=False,
                      help='Generate data.yaml file for YOLOv8 object detection training (default: False)')
    parser.add_argument('--images-dir', type=Path,
                      help='Base directory for images. Will be prepended to image paths from COCO json')
    parser.add_argument('--classification', action='store_true', default=False,
                      help='Process dataset for classification instead of object detection')
    parser.add_argument('--train-dir-name', type=str, default='train',
                      help='Name of the training directory (default: train)')
    parser.add_argument('--val-dir-name', type=str, default='val',
                      help='Name of the validation directory (default: val)')
    parser.add_argument('--test-dir-name', type=str, default='test',
                      help='Name of the test directory (default: test)')
    
    args = parser.parse_args()
    
    # If images_dir is not specified, use input_dir
    image_base_dir = args.images_dir if args.images_dir else args.input_dir
    
    print(f"\nInput directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Images directory: {image_base_dir}")
    print(f"Mode: {'Classification' if args.classification else 'Object Detection'}")
    
    if args.classification and args.include_data_yaml:
        print("Warning: --include-data-yaml is ignored in classification mode as YOLOv8 classification does not use data.yaml")
    
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
    
    # Create global category mapping first
    categories, category_id_to_index = create_global_category_mapping(found_splits)
    
    # Process each split
    splits = []
    for split_name, split_file in found_splits:
        splits.append(split_name)
        dir_name = {
            'train': args.train_dir_name,
            'val': args.val_dir_name,
            'test': args.test_dir_name
        }[split_name]
        
        if args.classification:
            split_categories = process_classification_split(
                split_file,
                args.output_dir,
                split_name,
                image_base_dir,
                dir_name
            )
        else:
            split_categories = process_split(
                split_file,
                args.output_dir,
                split_name,
                image_base_dir,
                dir_name,
                category_id_to_index,
                include_empty=args.include_empty
            )
    
    if args.include_data_yaml and categories and not args.classification:
        create_yaml_file(args.output_dir, categories, splits, args.train_dir_name, args.val_dir_name, args.test_dir_name)
    
    print("\nConversion complete!")
    print(f"Dataset created at: {args.output_dir}")
    
    if categories:
        print("\nFinal Class Mapping Summary:")
        print("---------------------------")
        print("YOLO ID | Original ID | Class Name")
        print("---------------------------")
        
        # Create a mapping of YOLO index to all original IDs that map to it
        index_to_originals = {}
        
        # Collect categories from all splits
        for split_name, split_file in found_splits:
            with open(split_file, 'r') as f:
                split_data = json.load(f)
                for cat in split_data['categories']:
                    yolo_idx = category_id_to_index[cat['id']]
                    if yolo_idx not in index_to_originals:
                        index_to_originals[yolo_idx] = set()
                    index_to_originals[yolo_idx].add((cat['id'], cat['name']))
        
        # Print sorted by YOLO index
        for yolo_idx in sorted(index_to_originals.keys()):
            originals = sorted(index_to_originals[yolo_idx])  # Convert set to sorted list
            # Print first one normally
            print(f"{yolo_idx:7d} | {originals[0][0]:10d} | {originals[0][1]}")
            # Print any duplicates with same YOLO ID
            for orig_id, name in originals[1:]:
                print(f"{' ':7s} | {orig_id:10d} | {name} (duplicate)")
        print("---------------------------")

if __name__ == '__main__':
    main()
