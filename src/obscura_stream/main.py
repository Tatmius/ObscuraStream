"""Main entry point for ObscuraStream application."""

import cv2
from obscura_stream.utils import (
    open_camera,
    apply_blur,
    VirtualCam,
    send_to_virtualcam,
    draw_box,
    draw_detection_list
)
from obscura_stream.detection import CombinedDetector


def main():
    """Main execution function."""
    cap = open_camera()
    detector = CombinedDetector()
    
    with VirtualCam() as cam:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detection = detector.detect(frame)
            
            if detection is None:
                detection_texts = ["No detection."]
            else:
                x1, y1, x2, y2 = detection.bbox
                apply_blur(frame, x1, y1, x2, y2)
                draw_box(frame, x1, y1, x2, y2, color=(0,255,0), text=detection.display_text)
                detection_texts = [detection.display_text]

            draw_detection_list(frame, detection_texts)
            cv2.imshow("ObscuraStream", frame)
            send_to_virtualcam(cam, frame)

            if cv2.waitKey(1) & 0xFF in [13, 27]:  # Enter or ESC
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main() 