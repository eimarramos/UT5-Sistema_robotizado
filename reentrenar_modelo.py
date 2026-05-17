import argparse
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reentrenar modelo YOLOv5 (repo oficial) con tu dataset local."
    )
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
    parser.add_argument(
        "--workers",
        type=int,
        default=0 if platform.system().lower() == "windows" else 8,
        help="Numero de workers del dataloader (Windows recomendado: 0-2)",
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
        default="",
        help="Dispositivo YOLOv5 (ej: 0, cpu). Vacio = automatico",
    )
    return parser.parse_args()


def resolve_yolov5_dir(yolov5_dir_arg: str) -> Path:
    candidates = [
        Path(yolov5_dir_arg),
        Path("yolov5"),
    ]

    for candidate in candidates:
        train_script = candidate / "train.py"
        if train_script.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "No se encontro YOLOv5 local. Clona el repo en 'third_party/yolov5' o pasa --yolov5-dir."
    )


def resolve_model(path_str: str) -> str:
    model_path = Path(path_str)
    if model_path.exists():
        return str(model_path.resolve())

    # YOLOv5 puede descargar automaticamente pesos oficiales si se pasa el nombre.
    return path_str


def build_absolute_dataset_yaml(data_yaml_path: Path) -> Path:
    raw = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8")) or {}
    base_dir = data_yaml_path.parent.resolve()

    def resolve_split_path(split_value: str) -> Path:
        split_path = Path(split_value)
        if split_path.is_absolute():
            return split_path

        # Caso normal: rutas relativas al directorio donde vive data.yaml
        candidate = (base_dir / split_path).resolve()
        if candidate.exists():
            return candidate

        # Fallback Roboflow común: "../train/images" dentro de dataset/data.yaml
        # que realmente apunta a dataset/train/images.
        trimmed = split_value
        while trimmed.startswith("../"):
            trimmed = trimmed[3:]
        if trimmed:
            candidate_trimmed = (base_dir / Path(trimmed)).resolve()
            if candidate_trimmed.exists():
                return candidate_trimmed

        return candidate

    for split_key in ("train", "val", "test"):
        split_value = raw.get(split_key)
        if not split_value:
            continue

        split_path = resolve_split_path(str(split_value))
        raw[split_key] = str(split_path)

    raw["path"] = ""

    temp_dir = Path(tempfile.mkdtemp(prefix="yolov5_data_"))
    out_yaml = temp_dir / "data_abs.yaml"
    out_yaml.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return out_yaml


def main() -> None:
    args = parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"No existe el archivo de dataset: {data_yaml}")

    yolov5_dir = resolve_yolov5_dir(args.yolov5_dir)
    train_script = yolov5_dir / "train.py"
    model_path = resolve_model(args.model)
    data_yaml_abs = build_absolute_dataset_yaml(data_yaml.resolve())

    project_dir = Path("runs/detect").resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(train_script),
        "--data",
        str(data_yaml_abs),
        "--weights",
        model_path,
        "--epochs",
        str(args.epochs),
        "--img",
        str(args.imgsz),
        "--batch-size",
        str(args.batch),
        "--project",
        str(project_dir),
        "--name",
        args.name,
        "--exist-ok",
        "--workers",
        str(args.workers),
    ]

    if args.device:
        command.extend(["--device", args.device])

    print("Ejecutando entrenamiento YOLOv5:")
    print(" ".join(command))
    subprocess.run(command, check=True, cwd=Path.cwd())


if __name__ == "__main__":
    main()
