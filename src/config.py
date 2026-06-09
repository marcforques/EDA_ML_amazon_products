from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "amazon_products_sales.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "amazon_products_clean.csv"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "optimized_random_forest.pkl"
FEATURES_PATH = MODEL_DIR / "features.pkl"

RANDOM_STATE = 42

SALES_LABELS = ["low", "medium", "high", "very_high"]