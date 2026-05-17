import argparse
import platform
import time
from pathlib import Path

import cv2
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inferencia YOLO en tiempo real con camara USB integrada o externa."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/chess_nano_v5/weights/best.pt",
        help="Ruta del modelo .pt",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Indice de camara (0, 1, 2, ...)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Umbral minimo de confianza",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="Umbral IoU para NMS",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Tamano de entrada del modelo",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Ancho del stream de camara",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Alto del stream de camara",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "dshow", "msmf", "gstreamer", "any"],
        help="Backend de camara: auto, dshow, msmf, gstreamer o any",
    )
    parser.add_argument(
        "--yolov5-dir",
        type=str,
        default="third_party/yolov5",
        help="Ruta al repositorio local de YOLOv5",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Dispositivo para PyTorch (auto, cpu, cuda, cuda:0)",
    )
    return parser.parse_args()


def resolve_model_path(model_arg: str) -> str:
    model_path = Path(model_arg)
    if model_path.exists():
        return str(model_path.resolve())

    raise FileNotFoundError(
        f"No se encontro el modelo '{model_arg}'. Verifica la ruta al .pt entrenado."
    )


def resolve_yolov5_dir(yolov5_dir_arg: str) -> Path:
    candidates = [
        Path(yolov5_dir_arg),
        Path("yolov5"),
    ]

    for candidate in candidates:
        hubconf_path = candidate / "hubconf.py"
        if hubconf_path.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "No se encontro YOLOv5 local. Clona el repo en 'third_party/yolov5' o pasa --yolov5-dir."
    )


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def open_camera(camera_index: int, backend: str, width: int, height: int) -> cv2.VideoCapture:
    win = platform.system().lower() == "windows"

    if backend == "dshow":
        candidates = [cv2.CAP_DSHOW]
    elif backend == "msmf":
        candidates = [cv2.CAP_MSMF]
    elif backend == "gstreamer":
        candidates = [cv2.CAP_GSTREAMER]
    elif backend == "any":
        candidates = [cv2.CAP_ANY]
    else:
        # En Windows, DirectShow suele ser mas estable que MSMF para webcams USB.
        candidates = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if win else [cv2.CAP_ANY]

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
            print(f"Camara abierta con backend={api}")
            return cap

        cap.release()

    raise RuntimeError(
        f"No se pudo abrir la camara con indice {camera_index}. "
        "Prueba otro valor con --camera o cambia --backend."
    )


def main() -> None:
    args = parse_args()

    yolov5_dir = resolve_yolov5_dir(args.yolov5_dir)
    model_path = resolve_model_path(args.model)
    device = resolve_device(args.device)

    print(f"Modelo cargado: {model_path}")
    print(f"Repo YOLOv5: {yolov5_dir}")
    print(f"Dispositivo: {device}")

    model = torch.hub.load(
        str(yolov5_dir),
        "custom",
        path=model_path,
        source="local",
        device=device,
    )
    model.conf = args.conf
    model.iou = args.iou

    cap = open_camera(args.camera, args.backend, args.width, args.height)

    print("Iniciando inferencia en tiempo real...")
    print("Teclas: q o ESC para salir")

    prev_time = time.time()
    consecutive_read_failures = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            consecutive_read_failures += 1
            if consecutive_read_failures <= 10:
                time.sleep(0.03)
                continue

            print("[AVISO] Fallo continuo de lectura. Reintentando abrir camara...")
            cap.release()
            cap = open_camera(args.camera, args.backend, args.width, args.height)
            consecutive_read_failures = 0
            continue

        consecutive_read_failures = 0

        # YOLOv5 via torch hub trabaja en RGB.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb, size=args.imgsz)

        rendered_rgb = results.render()[0]
        annotated = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)

        current_time = time.time()
        fps = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time = current_time

        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (30, 220, 30),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("YOLO - Tiempo Real", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
