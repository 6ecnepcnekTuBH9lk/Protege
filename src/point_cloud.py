from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils import ensure_dir


def disparity_to_depth_mm(
    disparity: np.ndarray,
    focal_px: float,
    baseline_mm: float,
    dmin_px: float = 0.0,
) -> np.ndarray:
    """
    Преобразует disparity в глубину по формуле Z = f * B / d.

    dmin_px добавляется для приближённого восстановления абсолютной disparity
    Middlebury после crop-операций.
    """
    disp_abs = disparity.astype(np.float32) + float(dmin_px)
    valid = np.isfinite(disp_abs) & (disp_abs > 0.1)

    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    depth[valid] = float(focal_px) * float(baseline_mm) / disp_abs[valid]
    return depth


def disparity_to_points(
    disparity: np.ndarray,
    image_bgr: np.ndarray,
    focal_px: float,
    baseline_mm: float,
    dmin_px: float = 0.0,
    max_depth_mm: float = 5000.0,
    max_points: int = 80000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Создаёт облако точек в системе координат левой камеры.

    X = (x - cx) * Z / f
    Y = (y - cy) * Z / f
    Z = f * B / disparity
    """
    h, w = disparity.shape[:2]
    cx = w / 2.0
    cy = h / 2.0

    depth = disparity_to_depth_mm(disparity, focal_px, baseline_mm, dmin_px=dmin_px)
    valid = np.isfinite(depth) & (depth > 0) & (depth < max_depth_mm)

    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    z = depth[ys, xs]
    x = (xs.astype(np.float32) - cx) * z / focal_px
    y = (ys.astype(np.float32) - cy) * z / focal_px

    points = np.column_stack([x, -y, z]).astype(np.float32)
    colors_bgr = image_bgr[ys, xs]
    colors_rgb = colors_bgr[:, ::-1].astype(np.uint8)

    if len(points) > max_points:
        # Детерминированная подвыборка, чтобы файлы PLY не были слишком тяжёлыми.
        idx = np.linspace(0, len(points) - 1, max_points).astype(int)
        points = points[idx]
        colors_rgb = colors_rgb[idx]

    return points, colors_rgb


def save_ply(path: Path, points: np.ndarray, colors_rgb: np.ndarray) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors_rgb):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
