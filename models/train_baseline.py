"""Train a baseline Logistic Regression spam classifier."""
from pathlib import Path
import joblib
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from data.preprocess import load_and_preprocess
from features.tfidf import fit_tfidf, save_vectorizer


def train(data_path: str, out_dir: str = "models/artifacts") -> dict:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df = load_and_preprocess(data_path)
    X_texts = df["message"].tolist()
    y = (df["label"] == "spam").astype(int).to_numpy()

    vec, X = fit_tfidf(X_texts)
    X = X
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # save artifacts
    model_path = Path(out_dir) / "logreg_baseline.joblib"
    vec_path = Path(out_dir) / "tfidf_vectorizer.joblib"
    joblib.dump(clf, model_path)
    save_vectorizer(vec, vec_path)

    report = {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "model_path": str(model_path),
        "vectorizer_path": str(vec_path),
    }
    # persist report for downstream tools / UI
    try:
        import json

        report_path = Path(out_dir) / "report.json"
        with open(report_path, "w") as fh:
            json.dump(report, fh)
    except Exception:
        # non-fatal: continue
        pass

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/sms_spam_no_header.csv")
    parser.add_argument("--out", default="models/artifacts")
    args = parser.parse_args()
    report = train(args.data, args.out)
    print("Training report:")
    for k, v in report.items():
        print(k, v)


if __name__ == "__main__":
    main()
