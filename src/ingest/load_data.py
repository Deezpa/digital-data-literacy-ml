import pandas as pd
from pathlib import Path

def load_csvs(input_dir: str) -> pd.DataFrame:
    p = Path(input_dir)
    dfs = [pd.read_csv(fp) for fp in p.glob("*.csv")]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

if __name__ == "__main__":
    df = load_csvs("data/raw")
    print(f"Loaded shape: {df.shape}")
