
import argparse, numpy as np, pandas as pd
from pathlib import Path
rng = np.random.default_rng(42)

def generate(n: int):
    age_band = rng.choice(["18-25","26-35","36-45","46+"], size=n, p=[0.25,0.35,0.25,0.15])
    region = rng.choice(["N","S","E","W"], size=n)
    literacy_baseline = rng.normal(55, 15, size=n).clip(0,100)
    module_hours = rng.gamma(2.0, 4.0, size=n)
    assessment_pre = (literacy_baseline + rng.normal(0,5,size=n)).clip(0,100)
    improvement = rng.normal(15, 10, size=n).clip(-10,40)
    assessment_post = (assessment_pre + improvement).clip(0,100)
    device_access = rng.integers(0,2,size=n)
    net_availability = rng.integers(0,2,size=n)
    income_band = rng.integers(1,5,size=n)
    dropout_prob = 1/(1+np.exp(-( -2 + 0.02*(60-assessment_pre) + 0.03*(10-module_hours) + 0.3*(1-device_access) )))
    dropout_flag = rng.binomial(1, dropout_prob)
    followup_90d = rng.binomial(1, 0.6 - 0.2*dropout_flag + 0.1*device_access).clip(0,1)
    target = dropout_flag
    df = pd.DataFrame(dict(
        age_band=age_band, region=region, literacy_baseline=literacy_baseline,
        module_hours=module_hours, assessment_pre=assessment_pre, assessment_post=assessment_post,
        improvement=improvement, device_access=device_access, net_availability=net_availability,
        income_band=income_band, dropout_flag=dropout_flag, followup_90d=followup_90d, target=target
    ))
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default="data/raw/ddl_synth.csv")
    args = ap.parse_args()
    df = generate(args.n)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote synthetic dataset: {args.out} shape={df.shape}")
