import cv2
import mediapipe as mp
import pyvirtualcam
from ultralytics import YOLO

# ----------------------------
# ユーザー設定：Webカメラ解像度
# ----------------------------
camera_width = 1280
camera_height = 720

# ----------------------------
# ユーザー設定：配信用（仮想カメラ用）解像度
# ----------------------------
stream_width = 640
stream_height = 360
stream_fps = 30

# 共通マージン設定
COMMON_MARGIN_RATIO = 0.15  # バウンディングボックスに対する割合（左右・下）
TOP_MARGIN_RATIO = 0.5      # 上方向だけ別途大きく取りたい場合

def open_camera(width, height, index=0):
    """
    指定した解像度でWebカメラをオープンし、実際の解像度を返す。
    """
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera opened with resolution: {actual_w} x {actual_h}")

    if not cap.isOpened():
        print("Could not open camera.")
        exit(1)

    return cap

def apply_blur(
    frame, 
    x1, y1, x2, y2, 
    top_ratio=TOP_MARGIN_RATIO, 
    common_ratio=COMMON_MARGIN_RATIO
):
    """
    指定した座標に基づき、上方向は top_ratio、
    それ以外は common_ratio でブラー範囲を拡張してブラーをかける。
    """
    ih, iw, _ = frame.shape
    box_w = x2 - x1
    box_h = y2 - y1

    # 左右・下のマージン
    margin_common = int(common_ratio * max(box_w, box_h))
    # 上方向のみ別途設定
    top_margin = int(top_ratio * max(box_w, box_h))

    x1_m = max(0, x1 - margin_common)
    y1_m = max(0, y1 - top_margin)
    x2_m = min(iw, x2 + margin_common)
    y2_m = min(ih, y2 + margin_common)

    roi = frame[y1_m:y2_m, x1_m:x2_m]
    blurred_roi = cv2.GaussianBlur(roi, (55, 55), 30)
    frame[y1_m:y2_m, x1_m:x2_m] = blurred_roi

def main():
    # カメラをオープン
    laptop_cam = open_camera(camera_width, camera_height)

    # MediaPipe Face Detectionのインスタンス
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    # YOLOのインスタンス作成（軽量モデルを推奨）
    yolo_model_path = "yolo11n.pt"  # 例：yolov8n
    yolo = YOLO(yolo_model_path)

    with pyvirtualcam.Camera(width=stream_width, height=stream_height, fps=stream_fps) as cam:
        print(f"Virtual camera created with resolution: {stream_width}x{stream_height}, fps: {stream_fps}")
        print("Press Enter or ESC to exit.")

        while True:
            ret, frame = laptop_cam.read()
            if not ret:
                print("Failed to read frame.")
                break

            # BGR→RGBに変換 (MediaPipeがRGBを期待)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                # 顔検出があった場合はブラーを適用
                ih, iw, _ = frame.shape
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x, y, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

                    # 座標をスケーリング
                    x1 = int(x * iw)
                    y1 = int(y * ih)
                    x2 = int((x + w) * iw)
                    y2 = int((y + h) * ih)

                    # ブラー処理を関数化
                    apply_blur(frame, x1, y1, x2, y2)

            else:
                # 顔が検出されなかった場合はYOLOに切り替え
                # device='cpu' を指定してCPU推論
                yolo_results = yolo.predict(frame, device='cpu')
                # 推論結果は yolo_results[0].boxes に含まれる
                for box in yolo_results[0].boxes:
                    cls_id = int(box.cls[0])  # クラスID
                    conf = box.conf[0]        # 信頼度
                    x1i, y1i, x2i, y2i = map(int, box.xyxy[0])

                    # "person"クラスのみ処理する
                    if yolo.names[cls_id] == "person":
                        # ここではブラーせずに枠描画を継続
                        # 必要に応じて apply_blur(frame, x1i, y1i, x2i, y2i) を呼び出す
                        cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
                        label = f"{yolo.names[cls_id]} {conf:.2f}"
                        cv2.putText(
                            frame, 
                            label, 
                            (x1i, y1i - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (0, 255, 0), 
                            1
                        )

            # OpenCVで表示
            cv2.imshow("Face Detection + Blur", frame)

            # 仮想カメラ用にリサイズ・RGB変換
            stream_frame = cv2.resize(frame, (stream_width, stream_height))
            stream_frame_rgb = cv2.cvtColor(stream_frame, cv2.COLOR_BGR2RGB)

            # pyvirtualcamに送信
            cam.send(stream_frame_rgb)
            cam.sleep_until_next_frame()

            # 30ミリ秒待機してEnter(13) or ESC(27)が押されたら終了
            key = cv2.waitKey(30) & 0xFF
            if key == 13 or key == 27:  # Enter or Esc
                break

        laptop_cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
