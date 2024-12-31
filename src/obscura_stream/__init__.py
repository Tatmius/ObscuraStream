"""ObscuraStream package initialization.

This package provides privacy-focused webcam streaming functionality
with face/person detection and automatic blur application.
"""

from obscura_stream.detection import CombinedDetector
from obscura_stream.utils import VirtualCam, apply_blur
from obscura_stream._version import __version__

__all__ = ["CombinedDetector", "VirtualCam", "apply_blur"]
