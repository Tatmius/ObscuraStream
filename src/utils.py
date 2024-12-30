import cv2
import pyvirtualcam
from config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, STREAM_WIDTH, STREAM_HEIGHT, STREAM_FPS
import time

def open_camera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS, index=0):
    """
    指定した解像度でWebカメラをオープンし、実際の解像度を返すcapを返却します。
    さらに、最初に一定枚数のフレームを読み込み、実測FPSを計測して出力します。
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

    # -- 実測FPSを計測する --
    n_frames = 30  # 計測するフレーム数
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

def apply_blur(frame, x1, y1, x2, y2, top_ratio=0.5, common_ratio=0.15):
    """
    x1,y1,x2,y2 の範囲にブラーをかける。
    top_ratioとcommon_ratioで上方向とその他のマージン量を調整可能。
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

def draw_box(frame, x1, y1, x2, y2, color=(0, 0, 255), thickness=2, text=None):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(frame, text, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            color, 1)
    
def draw_detection_list(frame, detection_texts):
    """
    detection_texts に格納された文字列を
    画面左上(10px, 20px)くらいから縦方向に描画する。
    """
    x_start, y_start = 10, 20
    line_height = 20  # 1行あたりの縦スペース
    
    for i, text in enumerate(detection_texts):
        y_pos = y_start + i * line_height
        cv2.putText(frame, text, (x_start, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1)


class VirtualCam:
    """
    pyvirtualcamをcontext managerとして使うためのラッパークラス。
    例:
        with VirtualCam(width=640, height=360, fps=30) as cam:
            # cam.send(...), cam.sleep_until_next_frame()
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
        # 特に明示的なクローズ処理は不要
        # （pyvirtualcamは __del__ 時に自動終了）
        pass

def send_to_virtualcam(cam, frame, target_width=STREAM_WIDTH, target_height=STREAM_HEIGHT):
    """
    仮想カメラに frame を送信する際のヘルパー関数。
      - リサイズ
      - BGR→RGB変換
      - cam.send()
      - cam.sleep_until_next_frame()
    をまとめています。
    """
    resized = cv2.resize(frame, (target_width, target_height))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    cam.send(rgb)
    cam.sleep_until_next_frame()
