
import argparse
import pandas as pd
from pathlib import Path

def make_features(input_csv: str, out_csv: str):
    df = pd.read_csv(input_csv)
    # Basic cleaning
    if "improvement" not in df.columns and {"assessment_pre","assessment_post"}.issubset(df.columns):
        df["improvement"] = df["assessment_post"] - df["assessment_pre"]
    # Simple encodings
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category").cat.codes
    # Drop any non-ML columns you don't want
    df = df.dropna()
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved features to {out_csv} (shape={df.shape})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    make_features(args.input, args.out)
