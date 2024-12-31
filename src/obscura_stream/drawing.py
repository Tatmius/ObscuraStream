"""Drawing utilities for ObscuraStream."""
import cv2
from obscura_stream.config import SHOW_BOX


def draw_box(frame, x1, y1, x2, y2, color=(0, 0, 255), thickness=2, text=None):
    """Draws a bounding box with optional text label on the frame.

    Args:
        frame (np.ndarray): Input image frame
        x1, y1 (int): Top-left corner coordinates
        x2, y2 (int): Bottom-right corner coordinates
        color (tuple): BGR color tuple. Defaults to (0, 0, 255) (red)
        thickness (int): Line thickness. Defaults to 2
        text (str, optional): Text to display above the box
    """
    if not SHOW_BOX:
        return

    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Draw text if provided
    if text:
        # Calculate text size for better positioning
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, font_thickness
        )
        
        # Draw text with background
        text_x = x1
        text_y = y1 - 5 if y1 > text_height + 10 else y1 + text_height + 10
        
        cv2.putText(
            frame, text,
            (text_x, text_y),
            font, font_scale,
            color, font_thickness
        )


def draw_detection_list(frame, detection_texts, start_pos=(10, 20)):
    """Draws a list of detection texts in the corner of the frame.

    Args:
        frame (np.ndarray): Input image frame
        detection_texts (list): List of strings to display
        start_pos (tuple): Starting position (x, y) for the text list.
                          Defaults to (10, 20)
    """
    if not SHOW_BOX or not detection_texts:
        return

    # Text drawing parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    line_height = 20
    color = (0, 255, 0)  # Green color for status text
    
    x_start, y_start = start_pos
    
    # Draw each line of text
    for i, text in enumerate(detection_texts):
        y_pos = y_start + i * line_height
        cv2.putText(
            frame, text,
            (x_start, y_pos),
            font, font_scale,
            color, font_thickness
        ) 