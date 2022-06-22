from pathlib import Path

import cv2
import numpy as np


def read_classes(file):
    classes = None
    with open(file, mode='r', encoding="utf-8") as f:
        classes = f.read().splitlines()
    return classes


# クラスの数だけカラーテーブルを生成する関数
def get_colors(num):
    colors = []
    np.random.seed(0)
    for i in range(num):
        color = np.random.randint(0, 256, [3]).astype(np.uint8)
        colors.append(color.tolist())
    return colors


def main(confidence_threshold=0.5, nms_threshold=0.4):
    directory = Path.cwd()
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise IOError("can't open capture!")

    size = (608, 608)
    weights = directory / 'models' / "yolov3-openimages.weights"
    config = directory / 'models' / "yolov3-openimages.cfg"

    model = cv2.dnn_DetectionModel(str(weights), str(config))
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # モデルの入力パラメーターを設定する
    scale = 1.0 / 255.0     # スケールファクター
    mean = (0.0, 0.0, 0.0)  # 差し引かれる平均値
    swap = True             # チャンネルの順番（True: RGB、False: BGR）
    crop = False            # クロップ
    model.setInputParams(scale, size, mean, swap, crop)

    # NMS（Non-Maximum Suppression）をクラスごとに処理する
    model.setNmsAcrossClasses(False)  # （True: 全体、False: クラスごと）

    classes = read_classes(str(directory / 'models' / "openimages.names"))
    colors = get_colors(len(classes))

    while True:
        result, image = capture.read()
        if not result:
            cv2.waitKey(0)
            break

        # 画像が3チャンネル以外の場合は3チャンネルに変換する
        channels = 1 if len(image.shape) == 2 else image.shape[2]
        if channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        class_ids, confidences, boxes = model.detect(
            image, confidence_threshold, nms_threshold
        )

        # 2次元配列(n, 1)から1次元配列(n, )に変換
        class_ids = np.array(class_ids).flatten()
        confidences = np.array(confidences).flatten()

        for class_id, confidence, box in zip(class_ids, confidences, boxes):
            class_name = classes[class_id]
            color = colors[class_id]
            thickness = 1
            cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)

            result = "{0} ({1:.3f})".format(class_name, confidence)
            point = (box[0], box[1] - 5)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            cv2.putText(
                image, result, point, font, scale,
                color, thickness, cv2.LINE_AA
            )

        cv2.imshow("object detection", image)
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
