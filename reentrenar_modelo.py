import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reentrenar modelo YOLO con tu dataset local")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Ruta a data.yaml")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov5n.pt",
        help="Modelo base .pt (recomendado en Nano: yolov5n.pt)",
    )
    parser.add_argument("--epochs", type=int, default=60, help="Numero de epocas")
    parser.add_argument("--imgsz", type=int, default=512, help="Tamano de imagen")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--name", type=str, default="chess_nano_v5", help="Nombre del experimento")
    return parser.parse_args()


def resolve_model(path_str: str) -> str:
    path = Path(path_str)
    if path.exists():
        return str(path)

    if path_str.lower() == "yolov5n.pt":
        return path_str

    fallback = Path("yolov5n.pt")
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
