import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reentrenar modelo YOLO con tu dataset local")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Ruta a data.yaml")
    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/chess_detector/weights/best.pt",
        help="Modelo base .pt (preentrenado o tu ultimo best.pt)",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Numero de epocas")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamano de imagen")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--name", type=str, default="chess_detector_custom", help="Nombre del experimento")
    return parser.parse_args()


def resolve_model(path_str: str) -> str:
    path = Path(path_str)
    if path.exists():
        return str(path)

    fallback = Path("yolo11n.pt")
    if fallback.exists():
        print(f"[AVISO] No existe {path}. Se usara {fallback}.")
        return str(fallback)

    raise FileNotFoundError(f"No existe {path} ni {fallback}.")


def main() -> None:
    args = parse_args()

    model_path = resolve_model(args.model)
    model = YOLO(model_path)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="runs/detect",
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
