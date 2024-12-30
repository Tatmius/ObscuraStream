# Face / Person Blur with MediaPipe & YOLO (Python)

このプログラムは、Web カメラから取得した映像に対して以下の処理を行い、リアルタイムで表示・仮想カメラ出力するスクリプトです。

1. **MediaPipe の Face Detection**で顔を検出し、検出領域にブラーをかける
2. 顔が検出されなかった場合（= 顔が小さい/後ろ向きなどの場合）は **YOLO** を使って人物を推定
3. ブラー適用や表示サイズの変更を行い、最終的に **pyvirtualcam** の仮想カメラとして出力
4. 配信ソフト（OBS 等）にこの仮想カメラを取り込むことで、**顔出しせずに配信**が可能

本プログラムは **「一人しか映らない環境」** を想定しており、複数人での検出には最適化していません。

---

## 目次

- [特徴](#特徴)
- [前提条件](#前提条件)
- [インストール](#インストール)
- [使い方](#使い方)
- [オプション／パラメータ](#オプションパラメータ)
- [留意点](#留意点)
- [ライセンス](#ライセンス)

---

## 特徴

- **軽量**: MediaPipe の Face Detection は高速・軽量で、CPU でもリアルタイム処理が狙いやすい
- **後頭部などで顔検出が失敗した場合**: YOLO を用いて人物領域を推定
- **ブラーの範囲調整**: 上方向に多めにマージンを取り、髪の毛までブラーするなど細かい設定が可能
- **仮想カメラ出力**: pyvirtualcam を通じて OBS などの配信ソフトに「カメラデバイス」として認識させ、容易に配信に取り込める

---

## 前提条件

- Python 3.8 以上推奨
- Windows / macOS / Linux いずれでも動作を想定（ただし pyvirtualcam は OS によって設定が異なる場合があります）
- CPU のみでも動作可（GPU があれば高速化可能）

### 必要なライブラリ

- **OpenCV**
- **MediaPipe**
- **pyvirtualcam**
- **Ultralytics の YOLO** (軽量モデルを使用)

---

## インストール

```bash
pip install opencv-python mediapipe pyvirtualcam ultralytics
```

事前に、ffmpeg やシステムのカメラ関連ドライバが正常に導入されていることを確認してください。

## 使い方

1. **Web カメラを接続**
2. **スクリプトを実行**

   ```bash
   python main.py
   ```

3. 仮想カメラが作成される（起動時にターミナル等へログが表示されます）
4. OBS などの配信ソフトを起動し、

- ソースを追加 → 映像キャプチャデバイス → 「Virtual Camera」や「pyvirtualcam」などの名称を選ぶ
- すると、ブラー処理済みの映像が入力されます

5. 配信ソフト側で配信開始すると、視聴者にはブラー処理された映像が表示されます。

## オプション／パラメータ

- camera_width, camera_height: Web カメラの解像度指定
- stream_width, stream_height, stream_fps: 仮想カメラ出力の解像度・FPS
- COMMON_MARGIN_RATIO: 顔バウンディングボックスの左右・下方向にかけるマージン比率
- TOP_MARGIN_RATIO: 上方向だけ別途設定するマージン比率
- YOLO モデル: 軽量モデル（例: yolov8n.pt）を使用すると CPU で動作させる際にパフォーマンスを稼ぎやすい

#### 例: main.py 内の設定

```

camera_width = 1280
camera_height = 720
stream_width = 640
stream_height = 360
stream_fps = 30

COMMON_MARGIN_RATIO = 0.15
TOP_MARGIN_RATIO = 0.5

```

## 留意点

1. **一人しか映っていない状況**を想定
   - 複数人が映る場合、検出できない人がいたり、どの人をブラーするか制御が必要になるかもしれません
2. **顔が極端に小さい or 後頭部だけ**などの場合、Face Detection が失敗し、YOLO の結果に頼ることになります
3. **仮想カメラ**が OS ごとに挙動や権限周りが異なる場合があります
   - Linux なら v4l2loopback, Windows ならユーザーアカウント制御の許可が必要など、環境設定が必要な場合があります
4. **パフォーマンス**
   - CPU のみで高解像度を処理する場合、フレームレートが落ちることがあります
   - 余裕があれば GPU 環境（CUDA 等）を利用することで高速化が可能です

## ライセンス

MIT License

本プログラムは以下のオープンソースライブラリを使用しています：

- OpenCV (Apache 2 License)
- MediaPipe (Apache License 2.0)
- Ultralytics YOLO (AGPL-3.0 License)
- pyvirtualcam (GNU General Public License v2.0
  )

これらのライブラリは、それぞれのライセンスに従って使用されています。
