from __future__ import annotations

import cv2
import numpy as np


def _prepare_gray(left_bgr: np.ndarray, right_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

    # Небольшое выравнивание контраста помогает классическим stereo-методам.
    left_gray = cv2.equalizeHist(left_gray)
    right_gray = cv2.equalizeHist(right_gray)
    return left_gray, right_gray


def _valid_num_disparities(width: int, requested: int = 96) -> int:
    """
    OpenCV требует, чтобы numDisparities делилось на 16.
    Для ThirdSize Middlebury ширина около 400-465 px, 96 обычно подходит.
    """
    max_reasonable = max(16, (width // 2 // 16) * 16)
    num_disp = min(requested, max_reasonable)
    num_disp = max(16, (num_disp // 16) * 16)
    return num_disp


def compute_stereo_bm(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    """
    StereoBM — быстрый блочный метод сопоставления.
    Возвращает карту диспаритета float32 в пикселях.
    """
    left_gray, right_gray = _prepare_gray(left_bgr, right_bgr)
    num_disp = _valid_num_disparities(left_gray.shape[1], requested=96)

    stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=15)
    stereo.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)
    stereo.setPreFilterSize(9)
    stereo.setPreFilterCap(31)
    stereo.setTextureThreshold(10)
    stereo.setUniquenessRatio(15)
    stereo.setSpeckleWindowSize(80)
    stereo.setSpeckleRange(2)
    stereo.setDisp12MaxDiff(1)

    disp = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan
    return disp


def compute_stereo_sgbm(left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    """
    StereoSGBM — полу-глобальное сопоставление.
    Обычно даёт более плотную и устойчивую disparity map, чем StereoBM.
    """
    left_gray, right_gray = _prepare_gray(left_bgr, right_bgr)
    num_disp = _valid_num_disparities(left_gray.shape[1], requested=128)
    block_size = 5
    channels = 1

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * channels * block_size**2,
        P2=32 * channels * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=80,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    disp = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan
    return disp


def compute_disparity(method: str, left_bgr: np.ndarray, right_bgr: np.ndarray) -> np.ndarray:
    method = method.lower().strip()
    if method == "bm":
        return compute_stereo_bm(left_bgr, right_bgr)
    if method == "sgbm":
        return compute_stereo_sgbm(left_bgr, right_bgr)
    raise ValueError(f"Неизвестный метод: {method}")
