from pathlib import Path

# Корень проекта: stereo_middlebury_project/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MIDDLEBURY_DIR = DATA_DIR / "middlebury_2006"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Открытый архив Middlebury 2006: ThirdSize, 2 views, all scenes.
MIDDLEBURY_ALL_2VIEWS_URL = (
    "https://vision.middlebury.edu/stereo/data/scenes2006/"
    "ThirdSize/zip-2views/ALL-2views.zip"
)

ARCHIVE_PATH = RAW_DIR / "ALL-2views.zip"

# Для ThirdSize карты диспаритета делятся на 3.
MIDDLEBURY_SCALE_DIV = 3.0

# Параметры Middlebury 2005/2006 для full-size; для ThirdSize делим focal на 3.
FOCAL_LENGTH_FULL_PX = 3740.0
BASELINE_MM = 160.0

# Методы, которые сравниваем в статье.
METHODS = ["bm", "sgbm"]
