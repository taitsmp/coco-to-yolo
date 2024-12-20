#!/usr/bin/env python3

try:
    from coco_to_yolo.coco_to_yolo import main
except ImportError:
    # For development when running directly
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from coco_to_yolo.coco_to_yolo import main

if __name__ == '__main__':
    main()
