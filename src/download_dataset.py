from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

from .config import ARCHIVE_PATH, MIDDLEBURY_ALL_2VIEWS_URL, MIDDLEBURY_DIR, RAW_DIR


def download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and dst.stat().st_size > 0:
        print(f"[OK] Архив уже есть: {dst}")
        return

    print(f"[DOWNLOAD] {url}")
    print(f"[TO]       {dst}")
    urllib.request.urlretrieve(url, dst)
    print("[OK] Скачивание завершено")


def extract_zip(zip_path: Path, dst_dir: Path, overwrite: bool = False) -> None:
    if dst_dir.exists() and any(dst_dir.iterdir()) and not overwrite:
        print(f"[OK] Датасет уже распакован: {dst_dir}")
        return

    if overwrite and dst_dir.exists():
        shutil.rmtree(dst_dir)

    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"[EXTRACT] {zip_path} -> {dst_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst_dir)
    print("[OK] Распаковка завершена")


def download_middlebury(overwrite: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MIDDLEBURY_DIR.mkdir(parents=True, exist_ok=True)

    download_file(MIDDLEBURY_ALL_2VIEWS_URL, ARCHIVE_PATH)
    extract_zip(ARCHIVE_PATH, MIDDLEBURY_DIR, overwrite=overwrite)
