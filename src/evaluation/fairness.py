
import argparse
import pandas as pd
import numpy as np

def demographic_parity_difference(df, y_hat_col, group_col):
    rates = df.groupby(group_col)[y_hat_col].mean()
    return rates.max() - rates.min()

def tpr_fpr_by_group(df, y_true, y_pred, group_col):
    out = {}
    for g, sub in df.groupby(group_col):
        tp = ((sub[y_true]==1) & (sub[y_pred]==1)).sum()
        fn = ((sub[y_true]==1) & (sub[y_pred]==0)).sum()
        fp = ((sub[y_true]==0) & (sub[y_pred]==1)).sum()
        tn = ((sub[y_true]==0) & (sub[y_pred]==0)).sum()
        tpr = tp / (tp+fn) if (tp+fn)>0 else np.nan
        fpr = fp / (fp+tn) if (fp+tn)>0 else np.nan
        out[g] = {"TPR": tpr, "FPR": fpr}
    return pd.DataFrame(out).T

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="processed CSV with y_true=target and optional yhat_proba")
    ap.add_argument("--target", default="target")
    ap.add_argument("--group", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if "yhat_proba" not in df.columns:
        raise SystemExit("Expected column 'yhat_proba' with predicted probabilities.")
    df["yhat"] = (df["yhat_proba"] >= args.threshold).astype(int)

    dpd = demographic_parity_difference(df, "yhat", args.group)
    print(f"Demographic Parity Difference ({args.group}): {dpd:.4f}")
    print("TPR/FPR by group:")
    print(tpr_fpr_by_group(df, args.target, "yhat", args.group))
