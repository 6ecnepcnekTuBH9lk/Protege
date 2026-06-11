from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from eth3d_dataset import (
    load_eth3d_samples,
    read_pfm,
    read_calib,
    read_image_gray,
    read_image_color,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_num_disparities(value: int) -> int:
    """
    OpenCV требует, чтобы numDisparities делилось на 16.
    """
    value = max(16, value)
    return int(np.ceil(value / 16) * 16)


def compute_stereo_bm(left_gray: np.ndarray, right_gray: np.ndarray, num_disp: int) -> np.ndarray:
    stereo = cv2.StereoBM_create(
        numDisparities=num_disp,
        blockSize=15,
    )

    disp = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan

    return disp


def compute_stereo_sgbm(left_gray: np.ndarray, right_gray: np.ndarray, num_disp: int) -> np.ndarray:
    block_size = 5

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    disp = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan

    return disp


def save_point_cloud_preview(points, colors, out_path, title="Point cloud"):
    """
    points: numpy array shape (N, 3)
    colors: numpy array shape (N, 3), values in [0,255] or [0,1]
    out_path: path to output png
    """
    if points is None or len(points) == 0:
        print(f"[WARN] Empty point cloud, preview not saved: {out_path}")
        return

    pts = np.asarray(points)
    cols = np.asarray(colors)

    # оставляем только корректные точки
    mask = np.isfinite(pts).all(axis=1)
    pts = pts[mask]
    cols = cols[mask]

    if len(pts) == 0:
        print(f"[WARN] No valid points after filtering: {out_path}")
        return

    # убираем слишком далекие выбросы по Z
    z = pts[:, 2]
    z_min, z_max = np.percentile(z, [2, 98])
    mask = (z >= z_min) & (z <= z_max)
    pts = pts[mask]
    cols = cols[mask]

    # прореживание, чтобы график не был слишком тяжелым
    max_points = 50000
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]
        cols = cols[idx]

    # нормализация цветов
    if cols.max() > 1.0:
        cols = cols / 255.0

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        c=cols,
        s=0.5
    )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=-60)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[OK] Point cloud preview saved: {out_path}")


def compute_stereo_sgbm_bilateral(left_gray: np.ndarray, right_gray: np.ndarray, num_disp: int) -> np.ndarray:
    disp = compute_stereo_sgbm(left_gray, right_gray, num_disp)

    valid = np.isfinite(disp)
    filled = np.where(valid, disp, 0).astype(np.float32)

    filtered = cv2.bilateralFilter(
        filled,
        d=7,
        sigmaColor=20,
        sigmaSpace=20,
    )

    filtered[~valid] = np.nan
    return filtered


def normalize_disparity_for_save(disp: np.ndarray) -> np.ndarray:
    valid = np.isfinite(disp)

    if valid.sum() == 0:
        return np.zeros(disp.shape, dtype=np.uint8)

    values = disp[valid]
    vmin = np.percentile(values, 1)
    vmax = np.percentile(values, 99)

    if vmax <= vmin:
        vmax = vmin + 1.0

    norm = (disp - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0, 1)
    norm[~valid] = 0

    return (norm * 255).astype(np.uint8)


def save_disparity_visualization(disp: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    disp_uint8 = normalize_disparity_for_save(disp)
    color = cv2.applyColorMap(disp_uint8, cv2.COLORMAP_TURBO)

    cv2.imwrite(str(out_path), color)


def evaluate_disparity(pred: np.ndarray, gt: np.ndarray, mask_nocc: np.ndarray | None = None) -> dict:
    """
    Считает MAE, RMSE и bad-pixel метрики.
    bad1/bad2/bad4 — доля пикселей, где ошибка больше 1/2/4 px.
    """
    valid_gt = np.isfinite(gt)

    if mask_nocc is not None:
        valid_gt &= mask_nocc

    valid_pred = np.isfinite(pred)
    valid = valid_gt & valid_pred

    total_gt = int(valid_gt.sum())
    total_valid = int(valid.sum())

    if total_valid == 0:
        return {
            "pixels_gt": total_gt,
            "pixels_valid_pred": 0,
            "coverage_percent": 0.0,
            "mae": np.nan,
            "rmse": np.nan,
            "bad1_percent": np.nan,
            "bad2_percent": np.nan,
            "bad4_percent": np.nan,
        }

    err = np.abs(pred[valid] - gt[valid])

    return {
        "pixels_gt": total_gt,
        "pixels_valid_pred": total_valid,
        "coverage_percent": total_valid / total_gt * 100 if total_gt > 0 else 0.0,
        "mae": float(np.mean(err)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bad1_percent": float(np.mean(err > 1.0) * 100),
        "bad2_percent": float(np.mean(err > 2.0) * 100),
        "bad4_percent": float(np.mean(err > 4.0) * 100),
    }


def disparity_to_point_cloud(
    left_color_bgr: np.ndarray,
    disp: np.ndarray,
    calib: dict,
    mask: np.ndarray | None = None,
    max_points: int = 200_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Восстанавливает 3D-точки по формуле:
    Z = f * B / (disparity + doffs)

    Цвет берётся из левого изображения im0.
    """
    cam0 = calib["cam0"]

    fx = float(cam0[0, 0])
    fy = float(cam0[1, 1])
    cx = float(cam0[0, 2])
    cy = float(cam0[1, 2])

    baseline = float(calib.get("baseline", 1.0))
    doffs = float(calib.get("doffs", 0.0))

    h, w = disp.shape

    valid = np.isfinite(disp) & (disp > 0)

    if mask is not None:
        valid &= mask

    denom = disp + doffs
    valid &= denom > 0

    ys, xs = np.where(valid)

    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    # Ограничиваем число точек, чтобы PLY не получался слишком тяжёлым.
    if len(xs) > max_points:
        idx = np.linspace(0, len(xs) - 1, max_points).astype(int)
        xs = xs[idx]
        ys = ys[idx]

    d = disp[ys, xs]
    z = fx * baseline / (d + doffs)

    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy

    points = np.stack([x, y, z], axis=1).astype(np.float32)

    colors_bgr = left_color_bgr[ys, xs]
    colors_rgb = colors_bgr[:, ::-1].astype(np.uint8)

    return points, colors_rgb


def save_ply(points: np.ndarray, colors: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
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

        for p, c in zip(points, colors):
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


def process_eth3d(dataset_root: Path, output_root: Path, num_disp: int, limit: int | None) -> None:
    samples = load_eth3d_samples(dataset_root)

    if limit is not None and limit > 0:
        samples = samples[:limit]

    print(f"ETH3D samples: {len(samples)}")
    print(f"Output: {output_root}")

    methods = {
        "bm": compute_stereo_bm,
        "sgbm": compute_stereo_sgbm,
        "sgbm_bilateral": compute_stereo_sgbm_bilateral,
    }

    rows = []

    for sample in samples:
        print(f"\n[SCENE] {sample.scene}")

        left_gray = read_image_gray(sample.left_path)
        right_gray = read_image_gray(sample.right_path)
        left_color = read_image_color(sample.left_path)

        gt = read_pfm(sample.gt_disp_path)
        calib = read_calib(sample.calib_path)

        mask_img = cv2.imread(str(sample.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            raise FileNotFoundError(f"Не удалось прочитать маску: {sample.mask_path}")

        # ETH3D: 255 — пиксели, видимые в обеих камерах.
        mask_nocc = mask_img >= 250

        scene_out = output_root / sample.scene
        scene_out.mkdir(parents=True, exist_ok=True)

        for method_name, method_func in methods.items():
            print(f"  method: {method_name}")

            pred = method_func(left_gray, right_gray, num_disp)

            metrics_all = evaluate_disparity(pred, gt, mask_nocc=None)
            metrics_nocc = evaluate_disparity(pred, gt, mask_nocc=mask_nocc)

            save_disparity_visualization(
                pred,
                scene_out / f"{method_name}_disp.png",
            )

            points, colors = disparity_to_point_cloud(
                left_color_bgr=left_color,
                disp=pred,
                calib=calib,
                mask=mask_nocc,
                max_points=200_000,
            )

            save_ply(
                points,
                colors,
                scene_out / f"{method_name}_cloud.ply",
            )

            save_point_cloud_preview(
                points,
                colors,
                scene_out / f"{method_name}_point_cloud_preview.png",
                title=f"{sample.scene}: {method_name.upper()} point cloud",
            )

            rows.append(
                {
                    "dataset": "ETH3D",
                    "scene": sample.scene,
                    "method": method_name,
                    "num_disparities": num_disp,

                    "all_pixels_gt": metrics_all["pixels_gt"],
                    "all_pixels_valid_pred": metrics_all["pixels_valid_pred"],
                    "all_coverage_percent": metrics_all["coverage_percent"],
                    "all_mae": metrics_all["mae"],
                    "all_rmse": metrics_all["rmse"],
                    "all_bad1_percent": metrics_all["bad1_percent"],
                    "all_bad2_percent": metrics_all["bad2_percent"],
                    "all_bad4_percent": metrics_all["bad4_percent"],

                    "nocc_pixels_gt": metrics_nocc["pixels_gt"],
                    "nocc_pixels_valid_pred": metrics_nocc["pixels_valid_pred"],
                    "nocc_coverage_percent": metrics_nocc["coverage_percent"],
                    "nocc_mae": metrics_nocc["mae"],
                    "nocc_rmse": metrics_nocc["rmse"],
                    "nocc_bad1_percent": metrics_nocc["bad1_percent"],
                    "nocc_bad2_percent": metrics_nocc["bad2_percent"],
                    "nocc_bad4_percent": metrics_nocc["bad4_percent"],

                    "point_cloud_points": len(points),
                }
            )

    df = pd.DataFrame(rows)

    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "eth3d_metrics.csv"
    df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print(f"\n[OK] Метрики сохранены: {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(PROJECT_ROOT / "data" / "ETH3D"),
        help="Путь к папке ETH3D",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "eth3d"),
        help="Папка для результатов ETH3D",
    )

    parser.add_argument(
        "--num-disp",
        type=int,
        default=128,
        help="Количество диспаритетов для OpenCV. Будет округлено до кратного 16.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Сколько сцен обработать. 0 — все сцены.",
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)

    num_disp = make_num_disparities(args.num_disp)
    limit = args.limit if args.limit > 0 else None

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("ETH3D root:", dataset_root)
    print("num_disp:", num_disp)

    process_eth3d(
        dataset_root=dataset_root,
        output_root=output_root,
        num_disp=num_disp,
        limit=limit,
    )


if __name__ == "__main__":
    main()