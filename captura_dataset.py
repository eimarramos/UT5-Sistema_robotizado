import argparse
import ast
import random
import time
from pathlib import Path

import cv2


DEFAULT_CLASSES = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Captura facil de imagenes + etiquetas YOLO usando un recuadro fijo."
    )
    parser.add_argument("--camera", type=int, default=0, help="Indice de camara")
    parser.add_argument("--width", type=int, default=1280, help="Ancho de camara")
    parser.add_argument("--height", type=int, default=720, help="Alto de camara")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="dataset",
        help="Carpeta raiz del dataset (contiene train/ valid/ test)",
    )
    parser.add_argument(
        "--yaml",
        type=str,
        default="dataset/data.yaml",
        help="Ruta del data.yaml para leer nombres de clases",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.15,
        help="Porcentaje de muestras que van a valid",
    )
    parser.add_argument(
        "--box-w",
        type=float,
        default=0.24,
        help="Ancho relativo del recuadro (0-1)",
    )
    parser.add_argument(
        "--box-h",
        type=float,
        default=0.58,
        help="Alto relativo del recuadro (0-1)",
    )
    parser.add_argument(
        "--burst-size",
        type=int,
        default=30,
        help="Numero de fotos en cada rafaga",
    )
    parser.add_argument(
        "--burst-delay",
        type=float,
        default=0.08,
        help="Segundos entre fotos de una rafaga",
    )
    return parser.parse_args()


def load_class_names(yaml_path: Path) -> list[str]:
    if not yaml_path.exists():
        print(f"[AVISO] No se encontro {yaml_path}. Se usan clases por defecto.")
        return DEFAULT_CLASSES

    try:
        text = yaml_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            clean = line.strip()
            if clean.startswith("names:"):
                _, raw = clean.split(":", 1)
                parsed = ast.literal_eval(raw.strip())
                if isinstance(parsed, list) and parsed:
                    return [str(x) for x in parsed]
    except Exception as exc:
        print(f"[AVISO] No se pudieron leer clases de {yaml_path}: {exc}")

    print("[AVISO] Se usan clases por defecto.")
    return DEFAULT_CLASSES


def make_dirs(dataset_root: Path) -> None:
    for split in ("train", "valid"):
        (dataset_root / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_root / split / "labels").mkdir(parents=True, exist_ok=True)


def compute_center_box(w: int, h: int, rel_w: float, rel_h: float) -> tuple[int, int, int, int]:
    box_w = max(1, min(w, int(round(w * rel_w))))
    box_h = max(1, min(h, int(round(h * rel_h))))

    cx = w // 2
    cy = h // 2

    x1 = max(0, cx - box_w // 2)
    y1 = max(0, cy - box_h // 2)
    x2 = min(w - 1, x1 + box_w - 1)
    y2 = min(h - 1, y1 + box_h - 1)
    return x1, y1, x2, y2


def yolo_bbox_from_pixels(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[float, float, float, float]:
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    return xc / w, yc / h, bw / w, bh / h


def save_sample(
    frame,
    dataset_root: Path,
    split: str,
    class_id: int,
    class_name: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    unique = int(time.time() * 1000) % 1000
    stem = f"{class_name}_{timestamp}_{unique:03d}"

    img_path = dataset_root / split / "images" / f"{stem}.jpg"
    lbl_path = dataset_root / split / "labels" / f"{stem}.txt"

    h, w = frame.shape[:2]
    xc, yc, bw, bh = yolo_bbox_from_pixels(x1, y1, x2, y2, w, h)

    cv2.imwrite(str(img_path), frame)
    lbl_path.write_text(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")

    return img_path


def print_controls(class_names: list[str]) -> None:
    print("\n=== Controles ===")
    print("q o ESC: salir")
    print("ESPACIO: capturar muestra")
    print("b: rafaga (30 por defecto)")
    print("n: siguiente clase")
    print("p: clase anterior")
    print("l: listar clases en consola")
    print("0-9: seleccionar clase rapida (solo ids 0..9)")
    print("=================\n")

    print("Clases disponibles:")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")


def capture_once(
    frame,
    dataset_root: Path,
    valid_ratio: float,
    current_class: int,
    class_names: list[str],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> tuple[str, Path]:
    split = "valid" if random.random() < valid_ratio else "train"
    saved = save_sample(
        frame,
        dataset_root,
        split,
        current_class,
        class_names[current_class],
        x1,
        y1,
        x2,
        y2,
    )
    return split, saved


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    yaml_path = Path(args.yaml)

    class_names = load_class_names(yaml_path)
    make_dirs(dataset_root)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo abrir la camara {args.camera}. Prueba --camera 1 o 2."
        )

    print_controls(class_names)

    current_class = 0
    train_count = 0
    valid_count = 0

    window = "Captura Dataset YOLO"

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[AVISO] No se pudo leer frame de la camara.")
            break

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = compute_center_box(w, h, args.box_w, args.box_h)

        view = frame.copy()

        cv2.rectangle(view, (x1, y1), (x2, y2), (50, 220, 50), 2)
        cv2.putText(
            view,
            f"Clase: {current_class} - {class_names[current_class]}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (30, 220, 30),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            view,
            f"train: {train_count} | valid: {valid_count}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            view,
            "ESPACIO: foto | b: rafaga | n/p: clase | q: salir",
            (15, h - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(window, view)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        if key == ord("n"):
            current_class = (current_class + 1) % len(class_names)
        elif key == ord("p"):
            current_class = (current_class - 1) % len(class_names)
        elif key == ord("l"):
            for idx, name in enumerate(class_names):
                marker = "<--" if idx == current_class else ""
                print(f"{idx}: {name} {marker}")
        elif ord("0") <= key <= ord("9"):
            candidate = key - ord("0")
            if candidate < len(class_names):
                current_class = candidate
        elif key == 32:
            split, saved = capture_once(
                frame,
                dataset_root,
                args.valid_ratio,
                current_class,
                class_names,
                x1,
                y1,
                x2,
                y2,
            )
            if split == "train":
                train_count += 1
            else:
                valid_count += 1
            print(
                f"[OK] Guardada ({split}) clase={current_class}:{class_names[current_class]} -> {saved.name}"
            )
        elif key == ord("b"):
            burst_size = max(1, args.burst_size)
            delay = max(0.0, args.burst_delay)
            print(
                f"[INFO] Iniciando rafaga de {burst_size} fotos para clase={current_class}:{class_names[current_class]}"
            )
            for i in range(burst_size):
                ok_burst, frame_burst = cap.read()
                if not ok_burst:
                    print("[AVISO] Rafaga interrumpida: no se pudo leer frame.")
                    break

                h_b, w_b = frame_burst.shape[:2]
                bx1, by1, bx2, by2 = compute_center_box(w_b, h_b, args.box_w, args.box_h)
                split, saved = capture_once(
                    frame_burst,
                    dataset_root,
                    args.valid_ratio,
                    current_class,
                    class_names,
                    bx1,
                    by1,
                    bx2,
                    by2,
                )
                if split == "train":
                    train_count += 1
                else:
                    valid_count += 1
                print(f"[RAFAGA {i + 1}/{burst_size}] ({split}) -> {saved.name}")
                if delay > 0:
                    time.sleep(delay)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSesion finalizada. train={train_count}, valid={valid_count}")


if __name__ == "__main__":
    main()
