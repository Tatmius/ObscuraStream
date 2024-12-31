"""Camera handling utilities for ObscuraStream."""
import cv2
import time
from obscura_stream.config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS


def get_camera_list(max_tries=10):
    """Get list of available cameras using OpenCV.

    Args:
        max_tries (int): Maximum number of indices to try

    Returns:
        list: List of tuples containing (index, name) of available cameras
    """
    available_cameras = []
    for i in range(max_tries):
        print(f"try video capture {i}")
        temp_cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        print(f"videoCapture {i} finished")
        if temp_cap.isOpened():
            ret, _ = temp_cap.read()
            if ret:
                name = f"Camera {i}"
                print(f"Camera {i} is testing...")
                try:
                    backend = temp_cap.getBackendName()
                    if backend == "DSHOW":
                        temp_cap.release()
                        temp_cap = cv2.VideoCapture(i + cv2.CAP_DSHOW)
                        name = f"Camera {i} (DirectShow)"
                        print(f"Camera {i} is using DirectShow backend.")
                    elif backend == "MSMF":
                        temp_cap.release()
                        temp_cap = cv2.VideoCapture(i + cv2.CAP_MSMF)
                        name = f"Camera {i} (MSMF)"
                        print(f"Camera {i} is using MSMF backend.")
                    else:
                        print(f"Camera {i} is using {backend} backend.")
                except cv2.error as e:
                    print(f"Camera {i} error: {str(e)}")
                except Exception as e:
                    print(f"Camera {i} unexpected error: {str(e)}")
                available_cameras.append((i, name))
            temp_cap.release()
        else:
            print(f"Camera {i} is failed to videoCapture.")
    
    return available_cameras


def open_camera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS, index=0):
    """Opens and configures a webcam with specified parameters.

    Args:
        width (int): Desired camera width in pixels
        height (int): Desired camera height in pixels
        fps (int): Desired camera FPS
        index (int): Camera device index

    Returns:
        cv2.VideoCapture: Configured camera capture object

    Raises:
        RuntimeError: If camera cannot be opened
    """
    # List available physical camera devices
    print("\nChecking available physical camera devices:")

    
    # Get OpenCV camera list
    available_cameras = get_camera_list()
    if not available_cameras:
        print("  No cameras found!")
    else:
        print("  OpenCV devices:")
        for idx, name in available_cameras:
            print(f"    {idx}: {name}")
    print()

    # Open the requested camera
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {index}")

    # Try to get device name
    device_name = f"Camera {index}"
    for idx, name in available_cameras:
        if idx == index:
            device_name = name
            break

    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Opened {device_name} with resolution: {actual_w} x {actual_h}")

    # Measure actual FPS
    n_frames = 30
    start_time = time.time()
    read_count = 0

    while read_count < n_frames:
        ret, test_frame = cap.read()
        if not ret:
            break
        read_count += 1

    duration = time.time() - start_time
    measured_fps = read_count / duration if duration > 0 else 0
    print(f"Measured approx. {measured_fps:.2f} FPS after opening camera.")

    return cap


def switch_camera(current_index, direction='next', max_tries=10):
    """Switches to the next/previous available camera.

    Args:
        current_index (int): Current camera index
        direction (str): 'next' or 'prev' to specify direction
        max_tries (int): Maximum number of indices to try

    Returns:
        tuple: (new_cap, new_index) or (None, current_index) if no camera found
    """
    print("Switching camera...")
    if not hasattr(switch_camera, '_cached_cameras'):
        print("Creating camera cache...")
        switch_camera._cached_cameras = get_camera_list(max_tries)
    
    available_indices = [idx for idx, _ in switch_camera._cached_cameras]
    
    if not available_indices:
        return None, current_index
        
    # Find next/previous available index
    if direction == 'next':
        next_idx = current_index
        while True:
            next_idx = (next_idx + 1) % max_tries
            if next_idx in available_indices:
                break
            if next_idx == current_index:
                return None, current_index
        print(f"Switching to next camera: {next_idx}")
    else:  # prev
        next_idx = current_index
        while True:
            next_idx = (next_idx - 1) % max_tries
            if next_idx in available_indices:
                break
            if next_idx == current_index:
                return None, current_index
        print(f"Switching to previous camera: {next_idx}")

    # Open new camera
    new_cap = cv2.VideoCapture(next_idx)
    print(f"finished VideoCapture {next_idx}")
    new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    new_cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    
    if new_cap.isOpened():
        camera_name = dict(switch_camera._cached_cameras).get(next_idx, f"Camera {next_idx}")
        print(f"\nSwitched to camera {next_idx}: {camera_name}")
        actual_w = new_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = new_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"New resolution: {actual_w} x {actual_h}")
        return new_cap, next_idx
    
    return None, current_index 