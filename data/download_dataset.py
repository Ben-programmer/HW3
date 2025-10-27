"""Utilities to download datasets for the project."""
from pathlib import Path
from urllib.request import urlretrieve


DEFAULT_URL = "https://raw.githubusercontent.com/huanchen1107/2025ML-spamEmail/main/datasets/sms_spam_no_header.csv"


def download_dataset(dest: str = "data/raw/sms_spam_no_header.csv", url: str = DEFAULT_URL) -> Path:
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    # use urllib.request.urlretrieve to avoid adding requests dependency for simple download
    print(f"Downloading dataset from {url} to {dest_path}")
    urlretrieve(url, dest_path)
    return dest_path


if __name__ == "__main__":
    p = download_dataset()
    print("Saved to:", p)
