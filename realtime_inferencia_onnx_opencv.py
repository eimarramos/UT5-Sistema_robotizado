import argparse
import ast
import glob
import platform
import time
from pathlib import Path

import cv2
import numpy as np


DEFAULT_NAMES = [
    "bishop",
    "black-bishop",
    "black-king",
    "black-knight",
    "black-pawn",
    "black-queen",
    "black-rook",
    "white-bishop",
    "white-king",
    "white-knight",
    "white-pawn",
    "white-queen",
    "white-rook",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inferencia YOLOv5 ONNX con OpenCV DNN para Jetson Nano."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/chess_nano_v5/weights/best.onnx",
        help="Ruta del modelo .onnx exportado desde YOLOv5",
    )
    parser.add_argument("--camera", type=int, default=0, help="Indice de camara")
    parser.add_argument("--conf", type=float, default=0.4, help="Confianza minima")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU para NMS")
    parser.add_argument("--imgsz", type=int, default=416, help="Tamano de entrada")
    parser.add_argument("--width", type=int, default=1280, help="Ancho de camara")
    parser.add_argument("--height", type=int, default=720, help="Alto de camara")
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "dshow", "msmf", "gstreamer", "v4l2", "any"],
        help="Backend de camara",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Listar camaras disponibles y salir",
    )
    parser.add_argument(
        "--dnn-target",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Usar CUDA de OpenCV DNN o CPU",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="dataset/data.yaml",
        help="YAML con nombres de clases",
    )
    return parser.parse_args()


def load_class_names(data_path):
    path = Path(data_path)
    if not path.exists():
        return DEFAULT_NAMES

    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("names:"):
            value = line.split(":", 1)[1].strip()
            try:
                names = ast.literal_eval(value)
                if isinstance(names, list) and names:
                    return [str(name) for name in names]
            except Exception:
                return DEFAULT_NAMES
    return DEFAULT_NAMES


def open_camera(camera_index, backend, width, height):
    win = platform.system().lower() == "windows"

    if backend == "dshow":
        candidates = [cv2.CAP_DSHOW]
    elif backend == "msmf":
        candidates = [cv2.CAP_MSMF]
    elif backend == "gstreamer":
        candidates = [cv2.CAP_GSTREAMER]
    elif backend == "v4l2":
        candidates = [cv2.CAP_V4L2]
    elif backend == "any":
        candidates = [cv2.CAP_ANY]
    else:
        candidates = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if win else [cv2.CAP_V4L2, cv2.CAP_GSTREAMER, cv2.CAP_ANY]

    for api in candidates:
        cap = cv2.VideoCapture(camera_index, api)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)

        ok, _ = cap.read()
        if ok:
            print("Camara abierta con backend={}".format(api))
            return cap

        cap.release()

    raise RuntimeError("No se pudo abrir la camara con indice {}".format(camera_index))


def list_cameras(max_index=10):
    win = platform.system().lower() == "windows"
    if win:
        backends = [
            ("any", cv2.CAP_ANY),
            ("dshow", cv2.CAP_DSHOW),
            ("msmf", cv2.CAP_MSMF),
        ]
    else:
        devices = sorted(glob.glob("/dev/video*"))
        if devices:
            print("Dispositivos encontrados:")
            for device in devices:
                print("  {}".format(device))
        else:
            print("No se encontraron dispositivos /dev/video*.")

        backends = [
            ("v4l2", cv2.CAP_V4L2),
            ("gstreamer", cv2.CAP_GSTREAMER),
            ("any", cv2.CAP_ANY),
        ]

    print("Probando indices de camara...")
    found = False
    for index in range(max_index):
        available = []
        for name, api in backends:
            cap = cv2.VideoCapture(index, api)
            ok = cap.isOpened()
            if ok:
                ret, _ = cap.read()
                ok = bool(ret)
            cap.release()
            if ok:
                available.append(name)

        if available:
            found = True
            print("  --camera {} funciona con backend(s): {}".format(index, ", ".join(available)))

    if not found:
        print("  No se encontro ninguna camara entre los indices 0 y {}.".format(max_index - 1))
        if not win:
            print("  Comprueba con: ls -l /dev/video*")
            print("  Si existe /dev/video1, prueba: --camera 1 --backend v4l2")


def letterbox(image, new_shape, color=(114, 114, 114)):
    shape = image.shape[:2]  # h, w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    ratio = min(float(new_shape[0]) / shape[0], float(new_shape[1]) / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2.0
    dh /= 2.0

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, ratio, (dw, dh)


def load_net(model_path, dnn_target):
    net = cv2.dnn.readNetFromONNX(str(model_path))
    if dnn_target == "cuda":
        cuda_devices = 0
        try:
            cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
        except Exception:
            cuda_devices = 0

        if cuda_devices < 1 or "NVIDIA CUDA" not in cv2.getBuildInformation():
            print("OpenCV DNN: CUDA no disponible en esta instalacion, usando CPU")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            return net

        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            print("OpenCV DNN: CUDA FP16")
        except Exception as exc:
            print("No se pudo activar CUDA en OpenCV DNN, usando CPU: {}".format(exc))
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("OpenCV DNN: CPU")
    return net


def postprocess(output, image_shape, ratio, pad, conf_thres, iou_thres):
    if isinstance(output, (list, tuple)):
        output = output[0]
    output = np.squeeze(output)

    if output.ndim == 1:
        output = np.expand_dims(output, 0)

    boxes = []
    scores = []
    class_ids = []
    image_h, image_w = image_shape[:2]

    for det in output:
        obj_conf = float(det[4])
        if obj_conf < conf_thres:
            continue

        class_scores = det[5:]
        class_id = int(np.argmax(class_scores))
        score = obj_conf * float(class_scores[class_id])
        if score < conf_thres:
            continue

        cx, cy, w, h = det[:4]
        x1 = (cx - w / 2.0 - pad[0]) / ratio
        y1 = (cy - h / 2.0 - pad[1]) / ratio
        x2 = (cx + w / 2.0 - pad[0]) / ratio
        y2 = (cy + h / 2.0 - pad[1]) / ratio

        x1 = max(0, min(int(x1), image_w - 1))
        y1 = max(0, min(int(y1), image_h - 1))
        x2 = max(0, min(int(x2), image_w - 1))
        y2 = max(0, min(int(y2), image_h - 1))
        boxes.append([x1, y1, max(0, x2 - x1), max(0, y2 - y1)])
        scores.append(score)
        class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
    if len(indices) == 0:
        return []

    detections = []
    for idx in np.array(indices).flatten():
        detections.append((boxes[idx], scores[idx], class_ids[idx]))
    return detections


def draw_detections(frame, detections, names):
    for box, score, class_id in detections:
        x, y, w, h = box
        label = names[class_id] if class_id < len(names) else str(class_id)
        text = "{} {:.2f}".format(label, score)
        color = (30, 220, 30)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, text, (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def main():
    args = parse_args()
    if args.list_cameras:
        list_cameras()
        return

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError("No se encontro el modelo ONNX: {}".format(model_path))

    names = load_class_names(args.data)
    print("Modelo ONNX: {}".format(model_path.resolve()))
    print("Clases: {}".format(len(names)))

    net = load_net(model_path, args.dnn_target)
    cap = open_camera(args.camera, args.backend, args.width, args.height)

    prev_time = time.time()
    print("Iniciando inferencia OpenCV DNN. Teclas: q o ESC para salir")

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.03)
            continue

        blob_img, ratio, pad = letterbox(frame, args.imgsz)
        blob = cv2.dnn.blobFromImage(blob_img, 1.0 / 255.0, (args.imgsz, args.imgsz), swapRB=True, crop=False)
        net.setInput(blob)
        output = net.forward()

        detections = postprocess(output, frame.shape, ratio, pad, args.conf, args.iou)
        annotated = draw_detections(frame, detections, names)

        current_time = time.time()
        fps = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time = current_time
        cv2.putText(annotated, "FPS: {:.1f}".format(fps), (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 220, 30), 2)

        cv2.imshow("YOLO ONNX OpenCV - Tiempo Real", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
