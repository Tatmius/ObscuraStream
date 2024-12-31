"""Masking and blur utilities for ObscuraStream."""
import cv2
import numpy as np
from obscura_stream.config import (
    BLUR_MARGIN_TOP, BLUR_MARGIN_COMMON, BLUR_KERNEL_SIZE,
    BLUR_SIGMA, FALLBACK_BLUR_KERNEL, FALLBACK_BLUR_SIGMA
)


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
    corner_radius=30
):
    """Applies Gaussian blur to a specified region with rounded corners.

    Args:
        frame (np.ndarray): Input image frame
        x1, y1, x2, y2 (int): Region boundaries
        top_ratio (float): Margin ratio for top direction
        common_ratio (float): Margin ratio for other directions
        kernel_size (int): Blur kernel size (must be odd)
        sigma (float): Blur sigma value
        corner_radius (int): Radius for rounded corners
    """
    ih, iw, _ = frame.shape
    box_w = x2 - x1
    box_h = y2 - y1

    # Calculate margins
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


def apply_full_frame_blur(
    frame,
    kernel_size=FALLBACK_BLUR_KERNEL,
    sigma=FALLBACK_BLUR_SIGMA
):
    """Applies a light Gaussian blur to the entire frame.

    Args:
        frame (np.ndarray): Input image frame
        kernel_size (int): Blur kernel size (must be odd)
        sigma (float): Blur sigma value

    Returns:
        np.ndarray: Blurred frame
    """
    kernel = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(frame, (kernel, kernel), sigma)


def load_mask_image(image_path, resize_method="contain"):
    """Loads and optionally resizes a mask image.

    Args:
        image_path (str): Path to the mask image
        resize_method (str): Resize method ("contain", "cover", "stretch")

    Returns:
        np.ndarray: Loaded image, or None if loading fails
    """
    if not image_path:
        return None
        
    try:
        mask_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if mask_img is None:
            print(f"Warning: Could not load mask image: {image_path}")
            return None
        return mask_img
    except Exception as e:
        print(f"Error loading mask image: {e}")
        return None


def apply_mask(
    frame,
    x1, y1, x2, y2,
    mask_type="blur",
    mask_image=None,
    top_ratio=BLUR_MARGIN_TOP,
    common_ratio=BLUR_MARGIN_COMMON,
    kernel_size=BLUR_KERNEL_SIZE,
    sigma=BLUR_SIGMA,
    corner_radius=30
):
    """Applies a mask (blur or image) to a specified region.

    Args:
        frame (np.ndarray): Input frame
        x1, y1, x2, y2 (int): Mask region boundaries
        mask_type (str): Mask type ("blur" or "image")
        mask_image (np.ndarray, optional): Image to use as mask
        Other parameters are the same as in apply_blur function
    """
    if mask_type == "blur" or mask_image is None:
        apply_blur(frame, x1, y1, x2, y2, top_ratio, common_ratio, 
                  kernel_size, sigma, corner_radius)
        return

    # Calculate margins
    ih, iw, _ = frame.shape
    box_w = x2 - x1
    box_h = y2 - y1
    margin_common = int(common_ratio * max(box_w, box_h))
    top_margin = int(top_ratio * max(box_w, box_h))

    x1_m = max(0, x1 - margin_common)
    y1_m = max(0, y1 - top_margin)
    x2_m = min(iw, x2 + margin_common)
    y2_m = min(ih, y2 + margin_common)

    # Mask region dimensions
    roi_h = y2_m - y1_m
    roi_w = x2_m - x1_m

    # Resize mask image
    mask_resized = cv2.resize(mask_image, (roi_w, roi_h))

    # Handle alpha channel
    if mask_resized.shape[2] == 4:
        alpha = mask_resized[:, :, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        rgb = mask_resized[:, :, :3]
        
        # Get original region
        roi = frame[y1_m:y2_m, x1_m:x2_m]
        
        # Alpha blending
        blended = (rgb * alpha + roi * (1 - alpha)).astype(np.uint8)
        frame[y1_m:y2_m, x1_m:x2_m] = blended
    else:
        frame[y1_m:y2_m, x1_m:x2_m] = mask_resized 