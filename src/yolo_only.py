import cv2
from ultralytics import YOLO

# 設定値やユーティリティ読込
from config import YOLO_MODEL_PATH
from utils import (
    open_camera,
    apply_blur,
    VirtualCam,
    send_to_virtualcam,
    draw_box,
    draw_detection_list
)

def main():
    cap = open_camera()
    yolo = YOLO(YOLO_MODEL_PATH)
    
    with VirtualCam() as cam:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLOで推論 (CPU)
            results = yolo.predict(frame, device="cpu", verbose=False)
            boxes = results[0].boxes

            # "person"クラスの候補を溜めるためのリスト
            person_candidates = []

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if yolo.names[cls_id] == "person":
                    # personだけリストに入れておく（枠描画やブラーは後で1つだけ）
                    person_candidates.append((conf, x1, y1, x2, y2))

            # 候補が1つもない場合は何もしない
            if len(person_candidates) == 0:
                # 画面左上にテキストだけ出す例
                detection_texts = ["No person detected."]
                draw_detection_list(frame, detection_texts)
            else:
                # confidenceが最大の1つを選ぶ
                best_conf, x1, y1, x2, y2 = max(person_candidates, key=lambda x: x[0])
                
                # 枠とブラーをかける
                apply_blur(frame, x1, y1, x2, y2, top_ratio=0.0, common_ratio=0.0)

                # 画面左上に表示するテキスト
                w = x2 - x1
                h = y2 - y1
                txt = f"person conf={best_conf:.2f} size={w}x{h}"
                
                # バウンディングボックスを描画
                draw_box(frame, x1, y1, x2, y2, color=(0,255,0), text=txt)

                # 複数行テキストを想定するためリストに
                detection_texts = [txt]
                draw_detection_list(frame, detection_texts)

            cv2.imshow("YOLO Only", frame)
            send_to_virtualcam(cam, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [13, 27]:  # Enter or ESC
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
