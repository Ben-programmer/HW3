"""Data preprocessing helper for spam dataset."""
from pathlib import Path
import pandas as pd
import re
from typing import Tuple


def load_dataset(path: str) -> pd.DataFrame:
    """Load CSV dataset and normalize column names to ['label','message']."""
    p = Path(path)
    df = pd.read_csv(p, header=None)
    # If file has header-like first row, try to detect
    # Common formats: 2 columns (label, message)
    if df.shape[1] == 2:
        df.columns = ["label", "message"]
    else:
        # fallback: try to read with header row
        df = pd.read_csv(p)
        cols = [c.lower() for c in df.columns]
        if "label" in cols and ("message" in cols or "text" in cols):
            # map
            label_col = df.columns[cols.index("label")]
            msg_col = df.columns[cols.index("message")] if "message" in cols else df.columns[cols.index("text")]
            df = df[[label_col, msg_col]]
            df.columns = ["label", "message"]
        else:
            raise ValueError("Unrecognized CSV schema; expected 2 columns (label,message) or header with label/message")
    return df


def clean_text(s: str) -> str:
    s = str(s)
    s = s.lower()
    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["message"] = df["message"].astype(str).map(clean_text)
    # normalize labels to 'spam'/'ham' if not
    df["label"] = df["label"].astype(str).str.lower().map(lambda x: "spam" if "spam" in x else "ham")
    return df[["label", "message"]]


def load_and_preprocess(path: str) -> pd.DataFrame:
    df = load_dataset(path)
    return preprocess_df(df)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sms_spam_no_header.csv"
    df = load_and_preprocess(path)
    print(df.head())
