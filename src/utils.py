import os
import random
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "Hinhve"

MODEL_PATH = MODELS_DIR / "best_dropout_pipeline.pkl"
RESULTS_CSV_PATH = OUTPUTS_DIR / "results_comparison.csv"
TUNING_CSV_PATH = OUTPUTS_DIR / "tuning_results.csv"
ERROR_CSV_PATH = OUTPUTS_DIR / "error_analysis.csv"


def get_first_csv_path() -> Path:
    """Tự động lấy file CSV đầu tiên trong thư mục data/raw/."""
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"Không tìm thấy file .csv nào trong '{DATA_DIR}'. "
            "Vui lòng đặt file dữ liệu sinh viên vào thư mục data/raw/."
        )
    return csv_files[0]


DATA_PATH = DATA_DIR / "data.csv"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_directories() -> None:
    for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def get_risk_tier(prob: float) -> str:
    if prob < 0.40:
        return "Low"
    elif prob < 0.70:
        return "Medium"
    return "High"