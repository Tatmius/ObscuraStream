# ObscuraStream

ObscuraStream is a privacy-focused webcam streaming tool that automatically detects and blurs faces and people in your video feed. Perfect for online meetings, streaming, or any situation where you want to maintain privacy while using your webcam.

## Features

- Automatic face detection with fallback to person detection
- Smooth, rounded-corner blur effect
- Configurable blur settings (strength, margins, etc.)
- Virtual camera output for OBS virtual camera
- Low latency processing

## Prerequisites

Before installing ObscuraStream, you need to have the following installed:

1. Python 3.12 or later
2. [OBS Virtual Camera](https://obsproject.com/) or any other virtual camera driver
3. [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip

### Install UV

For Windows users:

```powershell
# Install uv using PowerShell
curl -L --output uv.exe https://github.com/astral-sh/uv/releases/latest/download/uv-windows-x64.exe
```

For macOS/Linux users:

```bash
# Install uv using curl
curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-installer.sh | sh
```

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/ObscuraStream.git
cd ObscuraStream
```

2. Install the package:

```bash
uv pip install -e .
```

## Usage

1. Set up config.py based on your favor(Description below).
2. Run ObscuraStream:

```bash
uv run obscura_stream
```

3. Select pyvirtualCamera in OBS video source selection

## Configuration

All settings can be customized in `src/obscura_stream/config.py`:

### Camera Input Settings

```python
CAMERA_WIDTH = 1280    # Input camera resolution width (pixels)
CAMERA_HEIGHT = 720    # Input camera resolution height (pixels)
CAMERA_FPS = 24       # Input camera frame rate
```

Higher resolutions provide better quality but require more processing power.

### Virtual Camera Output Settings

```python
STREAM_WIDTH = 640     # Output stream resolution width (pixels)
STREAM_HEIGHT = 360    # Output stream resolution height (pixels)
STREAM_FPS = 24       # Output stream frame rate
```

Lower resolutions can improve performance while maintaining privacy.

### Detection Settings

```python
# Options: "hybrid" (face+person), "person_only"
DETECTION_MODE = "hybrid"

# Path to YOLO model weights (use smaller models for better performance)
YOLO_MODEL_PATH = "../models/yolo11s.pt"
```

### Visualization Settings

```python
SHOW_BOX = False     # Show/hide detection boxes and labels
```

### Blur Settings

```python
# Main blur parameters
BLUR_KERNEL_SIZE = 55        # Blur intensity (must be odd number)
BLUR_SIGMA = 30             # Blur spread (higher = more blur)
BLUR_MARGIN_TOP = 0.5       # Extra margin above detection (for hair)
BLUR_MARGIN_COMMON = 0.15   # Margin for other directions
BLUR_CORNER_RADIUS = 30     # Radius for rounded corners of blur

# Fallback blur (when no detection)
ENABLE_FALLBACK_BLUR = True  # Apply light blur when no face/person detected
FALLBACK_BLUR_KERNEL = 21    # Fallback blur intensity
FALLBACK_BLUR_SIGMA = 5      # Fallback blur spread
```

### Blur Parameters Explanation

The Gaussian blur effect is controlled by two main parameters:

- `BLUR_KERNEL_SIZE`: Determines the size of the Gaussian kernel. Higher values create stronger blur effects. Must be an odd number.
- Recommended range: 3 to 101
- Small values (e.g., 21) = Light blur
- Large values (e.g., 55) = Strong blur
- `BLUR_SIGMA`: Controls the standard deviation of the Gaussian function. Higher values spread the blur effect wider.
- Recommended range: 5 to 50
- Small values = Sharper blur transition
- Large values = Softer, more spread out blur

#### Tips

Kernel size and sigma values work together to create the final blur effect
For stronger privacy protection, use larger values for both parameters
For better performance, use smaller values
If the blur effect appears too harsh, try reducing the sigma value while keeping the kernel size

#### Blur Settings Explained

- **BLUR_MARGIN_TOP**: Larger value covers more area above the face (useful for hair)
- **BLUR_MARGIN_COMMON**: Controls blur extension around sides and bottom
- **BLUR_CORNER_RADIUS**: Higher values make blur edges rounder
- **Fallback Settings**: Apply a subtle blur when no face/person is detected

## Troubleshooting

### Common Issues

1. "Virtual camera not found"

   - Make sure OBS Virtual Camera is installed and running
   - Try restarting your computer after installing OBS

2. "No camera detected"

   - Check if your webcam is properly connected
   - Make sure no other application is using the camera

3. Low performance
   - Reduce input resolution (CAMERA_WIDTH, CAMERA_HEIGHT)
   - Lower BLUR_KERNEL_SIZE and BLUR_SIGMA values
   - Switch to "person_only" mode if face detection isn't necessary
   - Use a smaller YOLO model

## System Requirements

- GPU: Not required, but can improve performance if available
- Webcam: Any USB webcam or built-in camera

## License

MIT License

## Acknowledgments

- [MediaPipe](https://developers.google.com/mediapipe) for face detection
- [YOLOv8](https://docs.ultralytics.com/) for person detection
- [OBS Project](https://obsproject.com/) for virtual camera functionality
