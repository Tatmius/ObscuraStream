"""Main entry point for ObscuraStream application."""

import cv2
from obscura_stream.utils import (
    open_camera,
    apply_full_frame_blur,
    VirtualCam,
    send_to_virtualcam,
    draw_box,
    draw_detection_list,
    load_mask_image,
    apply_mask,
    switch_camera
)
from obscura_stream.detection import CombinedDetector
from obscura_stream.config import ENABLE_FALLBACK_BLUR, MASK_TYPE, MASK_IMAGE_PATH, MASK_IMAGE_RESIZE_METHOD


def main():
    """Main execution function."""
    current_camera_index = 0
    cap = open_camera(index=current_camera_index)
    detector = CombinedDetector()
    
    # マスク画像の読み込み
    mask_image = None
    if MASK_TYPE == "image" and MASK_IMAGE_PATH:
        mask_image = load_mask_image(MASK_IMAGE_PATH, MASK_IMAGE_RESIZE_METHOD)
        if mask_image is None:
            print("Warning: Falling back to blur mask")
    
    with VirtualCam() as cam:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detection = detector.detect(frame)
            
            if detection is None:
                detection_texts = ["No detection."]
                if ENABLE_FALLBACK_BLUR:
                    frame = apply_full_frame_blur(frame)
            else:
                x1, y1, x2, y2 = detection.bbox
                apply_mask(frame, x1, y1, x2, y2, 
                          mask_type=MASK_TYPE,
                          mask_image=mask_image)
                draw_box(frame, x1, y1, x2, y2, 
                        color=(0,255,0), 
                        text=detection.display_text)
                detection_texts = [detection.display_text]

            draw_detection_list(frame, detection_texts)
            cv2.imshow("ObscuraStream", frame)
            send_to_virtualcam(cam, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [13, 27]:  # Enter or ESC
                break
            elif key == ord('n'):  # 次のカメラ
                new_cap, new_index = switch_camera(current_camera_index, 'next')
                if new_cap is not None:
                    cap.release()
                    cap = new_cap
                    current_camera_index = new_index
            elif key == ord('p'):  # 前のカメラ
                new_cap, new_index = switch_camera(current_camera_index, 'prev')
                if new_cap is not None:
                    cap.release()
                    cap = new_cap
                    current_camera_index = new_index

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main() 