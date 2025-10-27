"""Train a baseline Logistic Regression spam classifier."""
from pathlib import Path
import joblib
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from data.preprocess import load_and_preprocess, preprocess_df
from features.tfidf import fit_tfidf, save_vectorizer


def train(
    data_path: str,
    out_dir: str = "models/artifacts",
    label_col: str | None = None,
    text_col: str | None = None,
    max_features: int = 10000,
    seed: int = 42,
    threshold: float = 0.5,
) -> dict:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # load dataset and optionally map custom columns
    if label_col is not None or text_col is not None:
        # try to read with header; if headers not present or columns missing, fall back
        try:
            df_raw = pd.read_csv(data_path)
            # default names if not provided
            lbl = label_col if label_col is not None else "label"
            txt = text_col if text_col is not None else "message"
            if lbl not in df_raw.columns or txt not in df_raw.columns:
                # fallback to automatic loader which handles headerless CSVs
                df = load_and_preprocess(data_path)
            else:
                df = df_raw[[lbl, txt]].copy()
                df.columns = ["label", "message"]
                df = preprocess_df(df)
        except Exception:
            df = load_and_preprocess(data_path)
    else:
        df = load_and_preprocess(data_path)

    X_texts = df["message"].tolist()
    y = (df["label"] == "spam").astype(int).to_numpy()

    vec, X = fit_tfidf(X_texts, max_features=max_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # use decision threshold on predicted probabilities if available
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_test)[:, 1]
        y_pred = (probs >= threshold).astype(int)
    else:
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
        "threshold": float(threshold),
        "max_features": int(max_features),
        "seed": int(seed),
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
    parser.add_argument("--label_col", default=None)
    parser.add_argument("--text_col", default=None)
    parser.add_argument("--max_features", default=10000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--threshold", default=0.5, type=float)
    args = parser.parse_args()
    report = train(
        args.data,
        args.out,
        label_col=args.label_col,
        text_col=args.text_col,
        max_features=args.max_features,
        seed=args.seed,
        threshold=args.threshold,
    )
    print("Training report:")
    for k, v in report.items():
        print(k, v)


if __name__ == "__main__":
    main()
