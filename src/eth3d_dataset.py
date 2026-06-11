from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import cv2


@dataclass
class ETH3DSample:
    scene: str
    left_path: Path
    right_path: Path
    calib_path: Path
    gt_disp_path: Path
    mask_path: Path


def read_pfm(path: Path) -> np.ndarray:
    """
    Читает PFM-карту диспаритета.
    В ETH3D disp0GT.pfm хранит float32-диспаритет для левого изображения.
    """
    with open(path, "rb") as f:
        header = f.readline().decode("ascii").rstrip()

        if header != "Pf":
            raise ValueError(f"Ожидался grayscale PFM с header='Pf', получено: {header}")

        dims_line = f.readline().decode("ascii").strip()
        while dims_line.startswith("#"):
            dims_line = f.readline().decode("ascii").strip()

        width, height = map(int, dims_line.split())

        scale = float(f.readline().decode("ascii").strip())
        endian = "<" if scale < 0 else ">"

        data = np.fromfile(f, endian + "f")
        data = np.reshape(data, (height, width))

        # PFM хранит строки снизу вверх
        data = np.flipud(data)

    return data.astype(np.float32)


def parse_matrix(value: str) -> np.ndarray:
    value = value.strip().strip("[]")
    rows = value.split(";")
    matrix = []

    for row in rows:
        nums = [float(x) for x in row.strip().split()]
        matrix.append(nums)

    return np.array(matrix, dtype=np.float32)


def read_calib(path: Path) -> dict:
    """
    Читает calib.txt в формате Middlebury v3.
    Основные поля:
    cam0, cam1, doffs, baseline, width, height, ndisp.
    """
    calib = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if value.startswith("["):
                calib[key] = parse_matrix(value)
            else:
                try:
                    if re.match(r"^-?\d+$", value):
                        calib[key] = int(value)
                    else:
                        calib[key] = float(value)
                except ValueError:
                    calib[key] = value

    return calib


def find_eth3d_root(dataset_root: Path) -> Path:
    """
    Поддерживает разные варианты распаковки ETH3D:

    1) data/ETH3D/two_view_training/<scene>/
       data/ETH3D/two_view_training_gt/<scene>/

    2) data/ETH3D/<scene>/
       где сцены лежат сразу внутри ETH3D
    """
    dataset_root = Path(dataset_root)

    candidates = [
        dataset_root,
        dataset_root / "ETH3D",
    ]

    for root in candidates:
        if not root.exists():
            continue

        # Вариант 1: классическая структура
        if (root / "two_view_training").exists():
            return root

        # Вариант 2: плоская структура, как у тебя
        for p in root.iterdir():
            if not p.is_dir():
                continue

            if (p / "im0.png").exists() and (p / "im1.png").exists():
                return root

    raise FileNotFoundError(
        "Не найдены сцены ETH3D. "
        f"Проверь путь: {dataset_root}"
    )


def load_eth3d_samples(dataset_root: str | Path) -> list[ETH3DSample]:
    dataset_root = Path(dataset_root)
    root = find_eth3d_root(dataset_root)

    # Вариант 1: если есть папки two_view_training и two_view_training_gt
    if (root / "two_view_training").exists():
        images_dir = root / "two_view_training"
        gt_dir = root / "two_view_training_gt"

    # Вариант 2: если сцены лежат сразу в data/ETH3D
    else:
        images_dir = root
        gt_dir = root

    samples = []

    for scene_dir in sorted(images_dir.iterdir()):
        if not scene_dir.is_dir():
            continue

        scene = scene_dir.name

        left_path = scene_dir / "im0.png"
        right_path = scene_dir / "im1.png"
        calib_path = scene_dir / "calib.txt"

        # ground truth может лежать либо в отдельной gt-папке, либо прямо в папке сцены
        gt_scene_dir = gt_dir / scene

        gt_disp_candidates = [
            gt_scene_dir / "disp0GT.pfm",
            scene_dir / "disp0GT.pfm",
        ]

        mask_candidates = [
            gt_scene_dir / "mask0nocc.png",
            scene_dir / "mask0nocc.png",
            gt_scene_dir / "mask0.png",
            scene_dir / "mask0.png",
        ]

        gt_disp_path = next((p for p in gt_disp_candidates if p.exists()), None)
        mask_path = next((p for p in mask_candidates if p.exists()), None)

        required = [
            left_path,
            right_path,
            calib_path,
            gt_disp_path,
            mask_path,
        ]

        missing = []
        for p in required:
            if p is None or not Path(p).exists():
                missing.append(str(p))

        if missing:
            print(f"[WARN] Пропуск сцены {scene}, нет файлов:")
            for p in missing:
                print("   ", p)
            continue

        samples.append(
            ETH3DSample(
                scene=scene,
                left_path=left_path,
                right_path=right_path,
                calib_path=calib_path,
                gt_disp_path=gt_disp_path,
                mask_path=mask_path,
            )
        )

    if not samples:
        raise RuntimeError(
            "Не найдено ни одной корректной сцены ETH3D. "
            "Проверь, что внутри папок сцен есть im0.png, im1.png, calib.txt, disp0GT.pfm и mask0nocc.png."
        )

    return samples


def read_image_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")

    return img


def read_image_color(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")

    return img