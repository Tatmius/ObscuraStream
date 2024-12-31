import cv2
import pyvirtualcam
from obscura_stream.config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, STREAM_WIDTH, STREAM_HEIGHT, STREAM_FPS, SHOW_BOX, BLUR_MARGIN_TOP, BLUR_MARGIN_COMMON, BLUR_KERNEL_SIZE, BLUR_SIGMA, FALLBACK_BLUR_KERNEL, FALLBACK_BLUR_SIGMA
import time
import numpy as np

def open_camera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS, index=0):
    """Opens and configures a webcam with specified parameters.

    Args:
        width (int): Desired camera width in pixels. Defaults to CAMERA_WIDTH.
        height (int): Desired camera height in pixels. Defaults to CAMERA_HEIGHT.
        fps (int): Desired camera FPS. Defaults to CAMERA_FPS.
        index (int): Camera device index. Defaults to 0.

    Returns:
        cv2.VideoCapture: Configured camera capture object.

    Raises:
        RuntimeError: If camera cannot be opened.
    """
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera opened with resolution: {actual_w} x {actual_h}")
    
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    # Measure actual FPS by capturing initial frames
    n_frames = 30
    start_time = time.time()
    read_count = 0

    while read_count < n_frames:
        ret, test_frame = cap.read()
        if not ret:
            break
        read_count += 1

    end_time = time.time()
    duration = end_time - start_time

    if duration > 0:
        measured_fps = read_count / duration
    else:
        measured_fps = 0

    print(f"Measured approx. {measured_fps:.2f} FPS after opening camera.")

    return cap

def create_rounded_rectangle_mask(width, height, radius):
    """Creates a rounded rectangle mask.

    Args:
        width (int): Mask width
        height (int): Mask height
        radius (int): Corner radius

    Returns:
        np.ndarray: Binary mask with rounded corners
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Fill center
    mask[radius:height-radius, :] = 255
    mask[:, radius:width-radius] = 255
    
    # Draw corners
    for corner_y, corner_x in [(radius, radius), 
                              (radius, width-radius-1),
                              (height-radius-1, radius),
                              (height-radius-1, width-radius-1)]:
        cv2.circle(mask, (corner_x, corner_y), radius, 255, -1)
    
    return mask

def apply_blur(
    frame,
    x1, y1, x2, y2,
    top_ratio=BLUR_MARGIN_TOP,
    common_ratio=BLUR_MARGIN_COMMON,
    kernel_size=BLUR_KERNEL_SIZE,
    sigma=BLUR_SIGMA,
    corner_radius=30  # 角の丸みの半径
):
    """Applies Gaussian blur to a specified region with rounded corners.

    Args:
        frame (np.ndarray): Input image frame.
        x1, y1, x2, y2 (int): Region boundaries.
        top_ratio (float): Margin ratio for top direction.
        common_ratio (float): Margin ratio for other directions.
        kernel_size (int): Blur kernel size (must be odd).
        sigma (float): Blur sigma value.
        corner_radius (int): Radius for rounded corners.
    """
    ih, iw, _ = frame.shape
    box_w = x2 - x1
    box_h = y2 - y1

    # Calculate margins for all directions
    # Use a larger ratio for top margin since the face detection bounding box
    # doesn't include hair - this ensures hair is properly masked
    margin_common = int(common_ratio * max(box_w, box_h))
    top_margin = int(top_ratio * max(box_w, box_h))

    x1_m = max(0, x1 - margin_common)
    y1_m = max(0, y1 - top_margin)
    x2_m = min(iw, x2 + margin_common)
    y2_m = min(ih, y2 + margin_common)

    # Extract region
    roi = frame[y1_m:y2_m, x1_m:x2_m].copy()
    roi_h, roi_w = roi.shape[:2]

    # Create rounded rectangle mask
    mask = create_rounded_rectangle_mask(roi_w, roi_h, corner_radius)
    
    # Apply blur
    kernel = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    blurred = cv2.GaussianBlur(roi, (kernel, kernel), sigma)

    # Blend original and blurred based on mask
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    result = (blurred * mask_3ch + roi * (1 - mask_3ch)).astype(np.uint8)
    
    # Put back the result
    frame[y1_m:y2_m, x1_m:x2_m] = result

def draw_box(frame, x1, y1, x2, y2, color=(0, 0, 255), thickness=2, text=None):
    """Draws a bounding box with optional text label on the frame.

    Args:
        frame (np.ndarray): Input image frame.
        x1, y1 (int): Top-left corner coordinates.
        x2, y2 (int): Bottom-right corner coordinates.
        color (tuple, optional): RGB color tuple. Defaults to (0, 0, 255).
        thickness (int, optional): Line thickness. Defaults to 2.
        text (str, optional): Text to display above the box. Defaults to None.
    """
    if SHOW_BOX:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, text, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 1)
    
def draw_detection_list(frame, detection_texts):
    """Draws a list of detection texts in the top-left corner of the frame.

    Args:
        frame (np.ndarray): Input image frame.
        detection_texts (list): List of strings to display.
    """
    if SHOW_BOX:
        x_start, y_start = 10, 20
        line_height = 20
        
        for i, text in enumerate(detection_texts):
            y_pos = y_start + i * line_height
            cv2.putText(frame, text, (x_start, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1)


class VirtualCam:
    """Context manager wrapper for pyvirtualcam.

    Provides a convenient way to create and manage virtual camera sessions.

    Example:
        ```python
        with VirtualCam(width=640, height=360, fps=30) as cam:
            cam.send(frame)
            cam.sleep_until_next_frame()
        ```
    """
    def __init__(self, width=STREAM_WIDTH, height=STREAM_HEIGHT, fps=STREAM_FPS):
        self.width = width
        self.height = height
        self.fps = fps
        self.cam = None

    def __enter__(self):
        self.cam = pyvirtualcam.Camera(
            width=self.width, 
            height=self.height, 
            fps=self.fps
        )
        print(f"Virtual camera created: {self.width}x{self.height}@{self.fps}fps")
        return self.cam

    def __exit__(self, exc_type, exc_val, exc_tb):
        # pyvirtualcam handles cleanup automatically in __del__
        pass

def send_to_virtualcam(cam, frame, target_width=STREAM_WIDTH, target_height=STREAM_HEIGHT):
    """Prepares and sends a frame to the virtual camera.

    Handles frame resizing and color space conversion before sending.

    Args:
        cam: Virtual camera instance.
        frame (np.ndarray): Input frame in BGR format.
        target_width (int, optional): Output width. Defaults to STREAM_WIDTH.
        target_height (int, optional): Output height. Defaults to STREAM_HEIGHT.
    """
    resized = cv2.resize(frame, (target_width, target_height))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    cam.send(rgb)
    cam.sleep_until_next_frame()

def apply_full_frame_blur(
    frame,
    kernel_size=FALLBACK_BLUR_KERNEL,
    sigma=FALLBACK_BLUR_SIGMA
):
    """Applies a light Gaussian blur to the entire frame.

    Args:
        frame (np.ndarray): Input image frame.
        kernel_size (int): Blur kernel size (must be odd).
        sigma (float): Blur sigma value.
    """
    # Ensure kernel size is odd
    kernel = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(frame, (kernel, kernel), sigma)
