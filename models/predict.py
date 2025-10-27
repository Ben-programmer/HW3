"""Minimal prediction script to load artifacts and return label+probability."""
import argparse
import joblib
from features.tfidf import load_vectorizer


def predict(text: str, model_path: str = "models/artifacts/logreg_baseline.joblib", vec_path: str = "models/artifacts/tfidf_vectorizer.joblib"):
    clf = joblib.load(model_path)
    vec = load_vectorizer(vec_path)
    X = vec.transform([text])
    prob = float(clf.predict_proba(X)[0][1]) if hasattr(clf, "predict_proba") else None
    label = int(clf.predict(X)[0])
    return {"label": "spam" if label == 1 else "ham", "probability": prob}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--model", default="models/artifacts/logreg_baseline.joblib")
    parser.add_argument("--vec", default="models/artifacts/tfidf_vectorizer.joblib")
    args = parser.parse_args()
    res = predict(args.text, args.model, args.vec)
    print(res)


if __name__ == "__main__":
    main()
