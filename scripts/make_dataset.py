import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH
from src.data_loader import load_csv
from src.cleaning import clean_dataset

def main():
    print("Cargando dataset raw...")
    df_raw = load_csv(RAW_DATA_PATH)

    print("Limpiando dataset...")
    df_clean = clean_dataset(df_raw)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    df_clean.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Dataset limpio guardado en: {PROCESSED_DATA_PATH}")
    print(f"Shape final: {df_clean.shape}")


if __name__ == "__main__":
    main()