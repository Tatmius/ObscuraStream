"""Utility functions for ObscuraStream."""
from obscura_stream.camera import get_camera_list, open_camera, switch_camera
from obscura_stream.virtual_camera import VirtualCam, send_to_virtualcam
from obscura_stream.masking import (
    apply_blur, apply_full_frame_blur,
    load_mask_image, apply_mask
)
from obscura_stream.drawing import draw_box, draw_detection_list

__all__ = [
    'get_camera_list', 'open_camera', 'switch_camera',
    'VirtualCam', 'send_to_virtualcam',
    'apply_blur', 'apply_full_frame_blur', 'load_mask_image', 'apply_mask',
    'draw_box', 'draw_detection_list'
]
