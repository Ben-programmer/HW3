"""TF-IDF feature extraction helpers."""
from pathlib import Path
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


def fit_tfidf(texts, max_features: int = 10000) -> Tuple[TfidfVectorizer, object]:
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    return vec, X


def save_vectorizer(vec: TfidfVectorizer, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, path)


def load_vectorizer(path: str) -> TfidfVectorizer:
    return joblib.load(path)


if __name__ == "__main__":
    print("TF-IDF helper module")
