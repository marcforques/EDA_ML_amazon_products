from pathlib import Path
import pandas as pd

def load_csv(path: str | Path):
    
    return pd.read_csv(path)