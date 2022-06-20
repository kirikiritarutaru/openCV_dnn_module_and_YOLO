# openCV_dnn_module_and_YOLO
OpenCVのDNNモジュールを用いて、YOLOv3 openimagesとScaled YOLOv4を動かすサンプルコード

# Prerequisites
- Python ライブラリをインストール
  - opencv-python
  - numpy
- `models`下に配置
  - YOLO-v3 openimages
    - [yolov3-openimages.weights](https://pjreddie.com/media/files/yolov3-openimages.weights)
    - [yolov3-openimages.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-openimages.cfg)
  - YOLOv4
    - YOLOv4
    - YOLOv4-tiny
    - YOLOv4x-mish
      - [yolov4x-mish.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4x-mish.weights)
      - [yolov4x-mish.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4x-mish.cfg)
    - YOLOv4-csp
    - YOLOv4-p5
    - YOLOv4-p6
      - [yolov4-p6.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-p6.weights)
      - [yolov4-p6.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-p6.cfg)
