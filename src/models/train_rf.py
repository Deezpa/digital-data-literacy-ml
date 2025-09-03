
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

def train_model(df: pd.DataFrame, target: str, outdir: str = "models"):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced")
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, proba)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, Path(outdir)/"rf.joblib")
    # Save feature list for serving
    pd.Series(X.columns).to_csv(Path(outdir)/"features.csv", index=False, header=False)
    # Persist predictions for fairness analysis
    test = X_test.copy()
    test["target"] = y_test.values
    test["yhat_proba"] = proba
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    test.to_csv("data/processed/ddl_features_test_with_preds.csv", index=False)
    print(f"AUC: {auc:.4f}")
    print(classification_report(y_test, (proba>0.5).astype(int)))

if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("data/processed/ddl_features.csv")
    train_model(df, target="target")
