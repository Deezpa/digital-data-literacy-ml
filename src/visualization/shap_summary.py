
import argparse
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", default="target")
    args = ap.parse_args()

    model = joblib.load(args.model)
    df = pd.read_csv(args.data)
    X = df.drop(columns=[args.target]) if args.target in df.columns else df

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    out = Path("reports/figures/shap_summary.png")
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved SHAP summary to {out}")
