from __future__ import annotations

import numpy as np


def disparity_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """
    Считает ошибки disparity относительно ground truth.

    bad1/bad2/bad4 — доля пикселей, где абсолютная ошибка больше 1/2/4 px.
    coverage — доля валидных пикселей среди всех валидных GT-пикселей.
    """
    gt_valid = np.isfinite(gt) & (gt > 0)
    valid = gt_valid & np.isfinite(pred) & (pred > 0)

    if not np.any(gt_valid):
        return {
            "valid_gt_pixels": 0,
            "valid_pred_pixels": 0,
            "coverage": 0.0,
            "mae": np.nan,
            "rmse": np.nan,
            "bad1": np.nan,
            "bad2": np.nan,
            "bad4": np.nan,
        }

    if not np.any(valid):
        return {
            "valid_gt_pixels": int(np.sum(gt_valid)),
            "valid_pred_pixels": 0,
            "coverage": 0.0,
            "mae": np.nan,
            "rmse": np.nan,
            "bad1": np.nan,
            "bad2": np.nan,
            "bad4": np.nan,
        }

    err = np.abs(pred[valid] - gt[valid])
    return {
        "valid_gt_pixels": int(np.sum(gt_valid)),
        "valid_pred_pixels": int(np.sum(valid)),
        "coverage": float(np.sum(valid) / np.sum(gt_valid)),
        "mae": float(np.mean(err)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bad1": float(np.mean(err > 1.0)),
        "bad2": float(np.mean(err > 2.0)),
        "bad4": float(np.mean(err > 4.0)),
    }


def error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    valid = np.isfinite(gt) & np.isfinite(pred) & (gt > 0) & (pred > 0)
    err = np.full(gt.shape, np.nan, dtype=np.float32)
    err[valid] = np.abs(pred[valid] - gt[valid])
    return err
