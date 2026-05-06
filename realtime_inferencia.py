import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inferencia YOLO en tiempo real con camara USB integrada o externa."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/chess_detector/weights/best.pt",
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
    return parser.parse_args()


def resolve_model_path(model_arg: str) -> str:
    model_path = Path(model_arg)
    if model_path.exists():
        return str(model_path)

    fallback = Path("yolo11n.pt")
    if fallback.exists():
        print(f"[AVISO] No se encontro '{model_arg}'. Se usara '{fallback}'.")
        return str(fallback)

    raise FileNotFoundError(
        f"No se encontro el modelo '{model_arg}' ni el fallback '{fallback}'."
    )


def main() -> None:
    args = parse_args()

    model_path = resolve_model_path(args.model)
    model = YOLO(model_path)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo abrir la camara con indice {args.camera}. "
            "Prueba otro valor con --camera."
        )

    print("Iniciando inferencia en tiempo real...")
    print("Teclas: q o ESC para salir")

    prev_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[AVISO] No se pudo leer frame de la camara.")
            break

        results = model.predict(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False,
        )

        annotated = results[0].plot()

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
