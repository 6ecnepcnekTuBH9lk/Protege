from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import MIDDLEBURY_DIR, MIDDLEBURY_SCALE_DIV, OUTPUT_DIR, PROJECT_ROOT
from .download_dataset import download_middlebury
from .pipeline import run_experiment
from .utils import find_middlebury_scenes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="StereoBM/StereoSGBM + 3D point cloud on Middlebury Stereo 2006"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Скачать и распаковать Middlebury 2006 ThirdSize ALL-2views.zip",
    )
    parser.add_argument(
        "--overwrite-data",
        action="store_true",
        help="Перезаписать распакованный датасет при скачивании",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Названия сцен для обработки, например: --scenes Aloe Cloth1 Flowerpots",
    )
    parser.add_argument(
        "--scale-div",
        type=float,
        default=MIDDLEBURY_SCALE_DIV,
        help="Коэффициент масштабирования disparity для ThirdSize; по умолчанию 3",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=80000,
        help="Максимальное число точек в каждом PLY-файле",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATASET_DIR: ", MIDDLEBURY_DIR)
    print("OUTPUT_DIR:  ", OUTPUT_DIR)

    if args.download:
        download_middlebury(overwrite=args.overwrite_data)

    scenes = find_middlebury_scenes(MIDDLEBURY_DIR)
    print(f"Найдено сцен: {len(scenes)}")
    if scenes:
        print("Сцены:", ", ".join(s.name for s in scenes))

    df = run_experiment(
        scenes=scenes,
        selected_scene_names=args.scenes,
        out_root=OUTPUT_DIR,
        scale_div=args.scale_div,
        max_points=args.max_points,
    )

    # Красивый вывод в консоль.
    cols = ["scene", "method", "mae", "rmse", "bad2", "coverage", "point_count"]
    existing_cols = [c for c in cols if c in df.columns]
    print("\n=== SUMMARY ===")
    print(df[existing_cols].to_string(index=False))
    print(f"\nГотово. Результаты сохранены в: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
