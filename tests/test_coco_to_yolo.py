import pytest
from pathlib import Path
from coco_to_yolo.coco_to_yolo import convert_bbox_coco_to_yolo

def test_basic_setup():
    """Basic test to verify testing infrastructure works"""
    assert True

def test_temp_directories(temp_dataset_dir, temp_output_dir):
    """Test that temporary directories are created properly"""
    assert temp_dataset_dir.parent.exists()
    assert temp_output_dir.parent.exists()

def test_create_test_image(temp_dataset_dir, create_test_image):
    """Test that we can create test images"""
    image_path = temp_dataset_dir / "images" / "test.jpg"
    created_path = create_test_image(image_path)
    assert created_path.exists()
    assert created_path.suffix == ".jpg" 