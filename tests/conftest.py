import pytest
from pathlib import Path
import json
import numpy as np
from PIL import Image

@pytest.fixture
def temp_dataset_dir(tmp_path):
    """Create a temporary directory for test datasets"""
    return tmp_path / "dataset"

@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary directory for test outputs"""
    return tmp_path / "output"

@pytest.fixture
def create_test_image():
    """Fixture to create test images with specified dimensions"""
    def _create_image(path: Path, width: int = 100, height: int = 100):
        # Create a simple test image using numpy and PIL
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JPEG
        img.save(path, 'JPEG')
        return path
    
    return _create_image

@pytest.fixture
def create_coco_annotation():
    """Fixture to create COCO format annotations"""
    def _create_annotation(path: Path, images=None, categories=None, annotations=None):
        coco_data = {
            "images": images or [],
            "categories": categories or [],
            "annotations": annotations or []
        }
        
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write COCO JSON file
        with open(path, 'w') as f:
            json.dump(coco_data, f)
        return path
    
    return _create_annotation 