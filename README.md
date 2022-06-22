# openCV_dnn_module_and_YOLO
OpenCVのDNNモジュールを用いて、USBカメラ映像に対して、YOLOv3 openimages, YOLOv4, Scaled YOLOv4を動かすサンプルコード

# Prerequisites
- Python ライブラリをインストール
  - opencv-python
  - numpy

- `models`下に`coco.names`と`openimages.names`を配置
  - [coco.names](https://github.com/AlexeyAB/darknet/blob/master/data/coco.names)
  - [openimages.names](https://github.com/AlexeyAB/darknet/blob/master/data/openimages.names)

- `models`下に各モデルのcfgとweightsを配置
  - YOLO-v3 openimages
    - [yolov3-openimages.weights](https://pjreddie.com/media/files/yolov3-openimages.weights)
    - [yolov3-openimages.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3-openimages.cfg)

  - YOLOv4
    - YOLOv4
      - [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)
      - [yolov4.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-csp.cfg)
    - YOLOv4-tiny
      - [yolov4-tiny.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights)
      - [yolov4-tiny.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg)

  - Scaled YOLOv4
    - YOLOv4x-mish
      - [yolov4x-mish.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4x-mish.weights)
      - [yolov4x-mish.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4x-mish.cfg)
    - YOLOv4-csp
      - [yolov4-csp.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-csp.weights)
      - [yolov4-csp.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-csp.cfg)
    - YOLOv4-p5
      - [yolov4-p5.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-p5.weights)
      - [yolov4-p5.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-p5.cfg)
    - YOLOv4-p6
      - [yolov4-p6.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-p6.weights)
      - [yolov4-p6.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-p6.cfg)

# 参考
- [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
- [ghmagazine/opencv_dl_book](https://github.com/ghmagazine/opencv_dl_book)
