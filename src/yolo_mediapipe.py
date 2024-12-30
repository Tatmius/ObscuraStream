import cv2
import mediapipe as mp
from ultralytics import YOLO

# 設定値やユーティリティ読込（config, utils 内の関数を仮定）
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
    # ---------------------------------
    # 1) カメラとモデルの準備
    # ---------------------------------
    cap = open_camera()
    yolo = YOLO(YOLO_MODEL_PATH)
    
    # MediaPipeのFace Detection
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(
        model_selection=0,         # 0: ショートレンジ, 1: フルレンジ (用途に応じて)
        min_detection_confidence=0.5
    )

    with VirtualCam() as cam:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 画像はBGR → MediaPipeはRGBを期待
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ---------------------------------
            # 2) MediaPipeで顔検出
            # ---------------------------------
            face_results = face_detector.process(rgb_frame)

            # 顔が見つかったかどうか
            if face_results.detections:
                # 複数の顔があり得るが、一人想定なので最もconfidenceが高い1つだけ
                best_face = None
                best_score = 0.0

                for detection in face_results.detections:
                    score = detection.score[0]  # confidence
                    if score > best_score:
                        best_score = score
                        best_face = detection

                # best_face からバウンディングボックスを取得
                if best_face is not None:
                    location = best_face.location_data
                    relative_box = location.relative_bounding_box
                    ih, iw, _ = frame.shape

                    x1 = int(relative_box.xmin * iw)
                    y1 = int(relative_box.ymin * ih)
                    w  = int(relative_box.width * iw)
                    h  = int(relative_box.height * ih)
                    x2 = x1 + w
                    y2 = y1 + h

                    # 枠 & ブラー
                    apply_blur(frame, x1, y1, x2, y2)
                    label = f"Face conf={best_score:.2f}"
                    draw_box(frame, x1, y1, x2, y2, color=(0,255,0), text=label)

                    # 左上に描画する用
                    detection_texts = [label]
                    draw_detection_list(frame, detection_texts)
                else:
                    # 万一ここに来るケースは少ないはず
                    detection_texts = ["No face, no YOLO used."]
                    draw_detection_list(frame, detection_texts)
            else:
                # ---------------------------------
                # 3) 顔が見つからなければYOLOでpersonを探す
                # ---------------------------------
                results = yolo.predict(frame, device="cpu", verbose=False)
                boxes = results[0].boxes

                person_candidates = []

                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if yolo.names[cls_id] == "person":
                        person_candidates.append((conf, x1, y1, x2, y2))

                if len(person_candidates) == 0:
                    # 誰もいない
                    detection_texts = ["No face, no person."]
                    draw_detection_list(frame, detection_texts)
                else:
                    # confidence が最も高いものを選ぶ
                    best_conf, x1, y1, x2, y2 = max(person_candidates, key=lambda x: x[0])
                    apply_blur(frame, x1, y1, x2, y2, top_ratio=0.0, common_ratio=0.0)
                    
                    w = x2 - x1
                    h = y2 - y1
                    txt = f"person conf={best_conf:.2f} size={w}x{h}"
                    draw_box(frame, x1, y1, x2, y2, color=(0,255,0), text=txt)

                    detection_texts = [txt]
                    draw_detection_list(frame, detection_texts)

            # 4) 表示＆仮想カメラ送信
            cv2.imshow("YOLO + MediaPipe Face Detection", frame)
            send_to_virtualcam(cam, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [13, 27]:  # Enter or ESC
                break

    # 後処理
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
