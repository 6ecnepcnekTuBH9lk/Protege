from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import BASELINE_MM, FOCAL_LENGTH_FULL_PX, METHODS, MIDDLEBURY_SCALE_DIV, OUTPUT_DIR
from .metrics import disparity_metrics, error_map
from .point_cloud import disparity_to_depth_mm, disparity_to_points, save_ply
from .stereo_methods import compute_disparity
from .utils import (
    StereoScene,
    ensure_dir,
    imread_bgr,
    imread_disparity,
    normalize_to_uint8,
    read_dmin,
    save_image,
)
from .visualize import (
    save_depth,
    save_disparity,
    save_epipolar_lines,
    save_error,
    save_point_cloud_preview,
    save_summary_charts,
)


def process_scene(
    scene: StereoScene,
    out_root: Path = OUTPUT_DIR,
    scale_div: float = MIDDLEBURY_SCALE_DIV,
    max_points: int = 80000,
) -> list[dict[str, float | str]]:
    scene_out = out_root / scene.name
    ensure_dir(scene_out)

    left_bgr = imread_bgr(scene.left_path)
    right_bgr = imread_bgr(scene.right_path)
    gt_disp = imread_disparity(scene.gt_disp_path, scale_div=scale_div)
    dmin_px = read_dmin(scene.dmin_path, scale_div=scale_div)

    # Для ThirdSize уменьшаем focal length вместе с изображением.
    focal_px = FOCAL_LENGTH_FULL_PX / scale_div

    save_image(scene_out / "input_left.png", left_bgr)
    save_image(scene_out / "input_right.png", right_bgr)
    save_epipolar_lines(scene_out / "epipolar_lines.png", left_bgr, right_bgr)
    save_disparity(scene_out / "gt_disparity.png", gt_disp)

    rows: list[dict[str, float | str]] = []

    for method in METHODS:
        disp = compute_disparity(method, left_bgr, right_bgr)

        metrics = disparity_metrics(disp, gt_disp)
        err = error_map(disp, gt_disp)
        depth = disparity_to_depth_mm(
            disp,
            focal_px=focal_px,
            baseline_mm=BASELINE_MM,
            dmin_px=dmin_px,
        )
        points, colors_rgb = disparity_to_points(
            disp,
            image_bgr=left_bgr,
            focal_px=focal_px,
            baseline_mm=BASELINE_MM,
            dmin_px=dmin_px,
            max_points=max_points,
        )

        prefix = method.lower()
        save_disparity(scene_out / f"{prefix}_disparity.png", disp)
        save_error(scene_out / f"{prefix}_error.png", err)
        save_depth(scene_out / f"{prefix}_depth.png", depth)
        save_ply(scene_out / f"{prefix}_point_cloud.ply", points, colors_rgb)
        save_point_cloud_preview(
            scene_out / f"{prefix}_point_cloud_preview.png",
            points,
            colors_rgb,
            title=f"{scene.name}: {method.upper()} point cloud",
        )

        # Сохраняем числовую disparity в npy, чтобы можно было переиспользовать в статье/графиках.
        np.save(scene_out / f"{prefix}_disparity.npy", disp)
        save_image(scene_out / f"{prefix}_disparity_gray.png", normalize_to_uint8(disp))

        row: dict[str, float | str] = {
            "scene": scene.name,
            "method": method.upper(),
            "dmin_px": dmin_px,
            "focal_px": focal_px,
            "baseline_mm": BASELINE_MM,
            "point_count": int(len(points)),
        }
        row.update(metrics)
        rows.append(row)

    return rows


def run_experiment(
    scenes: list[StereoScene],
    selected_scene_names: list[str] | None = None,
    out_root: Path = OUTPUT_DIR,
    scale_div: float = MIDDLEBURY_SCALE_DIV,
    max_points: int = 80000,
) -> pd.DataFrame:
    ensure_dir(out_root)

    if selected_scene_names:
        wanted = {x.lower() for x in selected_scene_names}
        scenes = [s for s in scenes if s.name.lower() in wanted]

    if not scenes:
        raise RuntimeError("Не найдено ни одной сцены для обработки. Проверь папку data/middlebury_2006.")

    all_rows: list[dict[str, float | str]] = []
    for scene in tqdm(scenes, desc="Middlebury scenes"):
        print(f"\n[SCENE] {scene.name}")
        rows = process_scene(
            scene,
            out_root=out_root,
            scale_div=scale_div,
            max_points=max_points,
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    summary_path = out_root / "summary_metrics.csv"
    df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    save_summary_charts(summary_path, out_root)
    return df
