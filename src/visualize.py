from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ensure_dir, normalize_to_uint8, save_image


def save_disparity(path: Path, disparity: np.ndarray) -> None:
    disp_u8 = normalize_to_uint8(disparity)
    color = cv2.applyColorMap(disp_u8, cv2.COLORMAP_TURBO)
    save_image(path, color)


def save_depth(path: Path, depth_mm: np.ndarray) -> None:
    # Для визуализации близкие объекты должны быть ярче.
    inv_depth = 1.0 / depth_mm
    depth_u8 = normalize_to_uint8(inv_depth)
    color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    save_image(path, color)


def save_error(path: Path, err: np.ndarray) -> None:
    err_u8 = normalize_to_uint8(err, vmin=0.0, vmax=10.0)
    color = cv2.applyColorMap(err_u8, cv2.COLORMAP_INFERNO)
    invalid = ~np.isfinite(err)
    color[invalid] = (0, 0, 0)
    save_image(path, color)


def save_epipolar_lines(path: Path, left_bgr: np.ndarray, right_bgr: np.ndarray, lines: int = 12) -> None:
    """
    Для rectified stereo-пары эпиполярные линии горизонтальные.
    Поэтому просто рисуем одинаковые горизонтальные линии на левом и правом изображениях.
    """
    h = min(left_bgr.shape[0], right_bgr.shape[0])
    w1 = left_bgr.shape[1]
    w2 = right_bgr.shape[1]

    left = left_bgr[:h].copy()
    right = right_bgr[:h].copy()

    for i, y in enumerate(np.linspace(20, h - 20, lines).astype(int)):
        # Цвета берём из colormap, чтобы линии не сливались.
        color = tuple(int(x) for x in cv2.applyColorMap(np.array([[i * 255 // max(1, lines - 1)]], dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0])
        cv2.line(left, (0, y), (w1 - 1, y), color, 1)
        cv2.line(right, (0, y), (w2 - 1, y), color, 1)

    combined = np.hstack([left, right])
    save_image(path, combined)


def save_point_cloud_preview(path: Path, points: np.ndarray, colors_rgb: np.ndarray, title: str) -> None:
    ensure_dir(path.parent)

    if len(points) == 0:
        fig = plt.figure(figsize=(7, 5))
        plt.title(f"{title}: no valid points")
        plt.axis("off")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    max_show = min(12000, len(points))
    idx = np.linspace(0, len(points) - 1, max_show).astype(int)
    pts = points[idx]
    cols = colors_rgb[idx].astype(np.float32) / 255.0

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], s=0.4, c=cols)
    ax.set_title(title)
    ax.set_xlabel("X, mm")
    ax.set_ylabel("Z, mm")
    ax.set_zlabel("Y, mm")
    ax.view_init(elev=18, azim=-70)

    # Убираем экстремальные выбросы из масштаба графика.
    for axis_idx, setter in [
        (0, ax.set_xlim),
        (2, ax.set_ylim),
        (1, ax.set_zlim),
    ]:
        vals = pts[:, axis_idx]
        lo, hi = np.percentile(vals, [2, 98])
        if hi > lo:
            setter(float(lo), float(hi))

    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_summary_charts(summary_csv: Path, out_dir: Path) -> None:
    if not summary_csv.exists():
        return

    df = pd.read_csv(summary_csv)
    if df.empty:
        return

    metrics = [
        ("mae", "Средняя абсолютная ошибка disparity, px", "metrics_mae.png"),
        ("rmse", "RMSE disparity, px", "metrics_rmse.png"),
        ("bad2", "Доля пикселей с ошибкой > 2 px", "metrics_bad2.png"),
        ("coverage", "Покрытие валидных пикселей", "metrics_coverage.png"),
    ]

    for metric, title, filename in metrics:
        pivot = df.pivot(index="scene", columns="method", values=metric)
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Сцена Middlebury")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)
