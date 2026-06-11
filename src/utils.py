from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class StereoScene:
    name: str
    scene_dir: Path
    left_path: Path
    right_path: Path
    gt_disp_path: Path
    dmin_path: Path | None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def imread_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
    return img


def imread_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
    return img


def imread_disparity(path: Path, scale_div: float) -> np.ndarray:
    """
    Читает ground-truth disparity Middlebury.

    В Middlebury 2006 ThirdSize значения disparity нужно делить на 3.
    Значение 0 означает неизвестную/невалидную диспаритетность.
    """
    disp = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if disp is None:
        raise FileNotFoundError(f"Не удалось прочитать disparity: {path}")

    if disp.ndim == 3:
        disp = cv2.cvtColor(disp, cv2.COLOR_BGR2GRAY)

    disp = disp.astype(np.float32) / float(scale_div)
    disp[disp <= 0] = np.nan
    return disp


def read_dmin(path: Path | None, scale_div: float) -> float:
    if path is None or not path.exists():
        return 0.0

    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return 0.0

    try:
        value = float(text.split()[0])
    except ValueError:
        return 0.0

    # Для ThirdSize приводим dmin к пикселям уменьшенного изображения.
    return value / float(scale_div)


def normalize_to_uint8(arr: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    data = arr.astype(np.float32).copy()
    finite = np.isfinite(data)
    if not np.any(finite):
        return np.zeros(data.shape, dtype=np.uint8)

    if vmin is None:
        vmin = float(np.nanpercentile(data[finite], 2))
    if vmax is None:
        vmax = float(np.nanpercentile(data[finite], 98))

    if abs(vmax - vmin) < 1e-6:
        vmax = vmin + 1.0

    data = np.clip((data - vmin) / (vmax - vmin), 0.0, 1.0)
    data[~finite] = 0.0
    return (data * 255).astype(np.uint8)


def save_image(path: Path, img: np.ndarray) -> None:
    ensure_dir(path.parent)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"Не удалось сохранить изображение: {path}")


def find_file_recursive(root: Path, filename: str) -> Path | None:
    direct = root / filename
    if direct.exists():
        return direct
    matches = list(root.rglob(filename))
    return matches[0] if matches else None


def find_middlebury_scenes(dataset_dir: Path) -> list[StereoScene]:
    """
    Находит сцены, где есть view1.png, view5.png, disp1.png.
    Работает даже если архив распаковался с промежуточными папками.
    """
    scenes: list[StereoScene] = []
    used_dirs: set[Path] = set()

    for left_path in dataset_dir.rglob("view1.png"):
        scene_dir = left_path.parent
        if scene_dir in used_dirs:
            continue

        right_path = scene_dir / "view5.png"
        gt_path = scene_dir / "disp1.png"
        dmin_path = scene_dir / "dmin.txt"

        # Иногда файлы могут лежать чуть выше/ниже — делаем мягкий поиск.
        if not right_path.exists():
            right_path = find_file_recursive(scene_dir, "view5.png") or right_path
        if not gt_path.exists():
            gt_path = find_file_recursive(scene_dir, "disp1.png") or gt_path

        if left_path.exists() and right_path.exists() and gt_path.exists():
            scenes.append(
                StereoScene(
                    name=scene_dir.name,
                    scene_dir=scene_dir,
                    left_path=left_path,
                    right_path=right_path,
                    gt_disp_path=gt_path,
                    dmin_path=dmin_path if dmin_path.exists() else None,
                )
            )
            used_dirs.add(scene_dir)

    scenes.sort(key=lambda s: s.name.lower())
    return scenes
