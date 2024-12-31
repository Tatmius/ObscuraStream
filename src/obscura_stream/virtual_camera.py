"""Virtual camera utilities for ObscuraStream."""
import cv2
import pyvirtualcam
from obscura_stream.config import STREAM_WIDTH, STREAM_HEIGHT, STREAM_FPS, VIRTUAL_CAM_DEVICE


class VirtualCam:
    """Context manager wrapper for pyvirtualcam.

    A context manager that creates and manages virtual camera sessions.
    Lists available virtual camera devices and uses either the specified device
    or the default device to create a virtual camera.

    Args:
        width (int): Output frame width in pixels. Defaults to STREAM_WIDTH.
        height (int): Output frame height in pixels. Defaults to STREAM_HEIGHT.
        fps (int): Frame rate. Defaults to STREAM_FPS.
        device (str): Name of the virtual camera device to use.
                     If empty string, uses the first available device.

    Example:
        ```python
        with VirtualCam(width=640, height=360, fps=30) as cam:
            cam.send(frame)
            cam.sleep_until_next_frame()
        ```

    Raises:
        Exception: If virtual camera creation fails.
    """
    def __init__(self, width=STREAM_WIDTH, height=STREAM_HEIGHT, fps=STREAM_FPS, device=VIRTUAL_CAM_DEVICE):
        self.width = width
        self.height = height
        self.fps = fps
        self.device = device
        self.cam = None

    def __enter__(self):
        try:
            # Try to list available devices (newer versions of pyvirtualcam)
            available_cameras = []
            try:
                if hasattr(pyvirtualcam.Camera, 'get_available_devices'):
                    available_cameras = pyvirtualcam.Camera.get_available_devices()
                    print("Available virtual camera devices:")
                    for i, cam in enumerate(available_cameras):
                        print(f"  {i}: {cam}")
            except:
                print("Note: Could not list virtual camera devices (older pyvirtualcam version)")

            if self.device:
                # Use specified device
                self.cam = pyvirtualcam.Camera(
                    width=self.width, 
                    height=self.height, 
                    fps=self.fps,
                    device=self.device
                )
                print(f"Using specified virtual camera device: {self.device}")
            else:
                # Use default device
                self.cam = pyvirtualcam.Camera(
                    width=self.width, 
                    height=self.height, 
                    fps=self.fps
                )
                print(f"Using default virtual camera device")

        except Exception as e:
            print(f"Error creating virtual camera: {e}")
            if available_cameras:
                print("Available devices were:", available_cameras)
            raise

        print(f"Virtual camera created: {self.width}x{self.height}@{self.fps}fps")
        return self.cam

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Performs cleanup of the virtual camera."""
        if self.cam:
            try:
                self.cam.close()
            except:
                pass


def send_to_virtualcam(cam, frame, target_width=STREAM_WIDTH, target_height=STREAM_HEIGHT):
    """Prepares and sends a frame to the virtual camera.

    Handles frame resizing and color space conversion before sending.

    Args:
        cam: Virtual camera instance
        frame (np.ndarray): Input frame in BGR format
        target_width (int): Output width
        target_height (int): Output height
    """
    resized = cv2.resize(frame, (target_width, target_height))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    cam.send(rgb)
    cam.sleep_until_next_frame() 