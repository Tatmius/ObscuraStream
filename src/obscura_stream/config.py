"""Camera and stream configuration parameters.

This module contains the configuration settings for:
    - Physical camera input (resolution and FPS)
    - Virtual camera output (resolution and FPS)
    - Detection preferences and model paths
    - Blur parameters
"""

# Camera input settings
CAMERA_WIDTH = 1280    # pixels
CAMERA_HEIGHT = 720    # pixels
CAMERA_FPS = 24       # frames per second

# Virtual camera output settings
STREAM_WIDTH = 640    # pixels
STREAM_HEIGHT = 360   # pixels
STREAM_FPS = 24      # frames per second

# Detection settings
DETECTION_MODE = "hybrid"  # Options: "hybrid" (face+person), "person_only"
YOLO_MODEL_PATH = "../models/yolo11s.pt"  # Path to YOLO model weights

# Visualization settings
SHOW_BOX = False     # Whether to show detection boxes and labels

# Blur settings
BLUR_KERNEL_SIZE = 55  # Must be odd number
BLUR_SIGMA = 30       # Higher value = more blur
BLUR_MARGIN_TOP = 0.5     # Margin ratio for top of detection
BLUR_MARGIN_COMMON = 0.15  # Margin ratio for left/right/bottom of detection