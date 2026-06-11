from pathlib import Path
import urllib.request
import py7zr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ETH3D_ROOT = PROJECT_ROOT / "data" / "ETH3D"

URLS = {
    "two_view_training.7z": "https://eth3d.ethz.ch/data/two_view_training.7z",
    "two_view_training_gt.7z": "https://eth3d.ethz.ch/data/two_view_training_gt.7z",
}


def download_file(url: str, out_path: Path) -> None:
    if out_path.exists():
        print(f"[SKIP] Уже скачано: {out_path}")
        return

    print(f"[DOWNLOAD] {url}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def progress(block_num, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        percent = min(downloaded / total_size * 100, 100)
        print(f"\r{percent:6.2f}%", end="")

    urllib.request.urlretrieve(url, out_path, progress)
    print(f"\n[OK] Скачано: {out_path}")


def extract_7z(archive_path: Path, extract_to: Path) -> None:
    print(f"[EXTRACT] {archive_path.name}")
    extract_to.mkdir(parents=True, exist_ok=True)

    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        archive.extractall(path=extract_to)

    print(f"[OK] Распаковано в: {extract_to}")


def main() -> None:
    ETH3D_ROOT.mkdir(parents=True, exist_ok=True)

    for filename, url in URLS.items():
        archive_path = ETH3D_ROOT / filename
        download_file(url, archive_path)
        extract_7z(archive_path, ETH3D_ROOT)

    print("\nГотово. Проверь структуру:")
    print(ETH3D_ROOT)
    print("  two_view_training/")
    print("  two_view_training_gt/")


if __name__ == "__main__":
    main()