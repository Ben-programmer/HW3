# HW3 — Spam Classifier (spec-driven)

This repository contains OpenSpec-driven change proposals and a small prototype for a spam classifier.

Quick start (Phase1 baseline):

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Download dataset and run baseline training:

```bash
python data/download_dataset.py
python models/train_baseline.py --data data/raw/sms_spam_no_header.csv --out models/artifacts
```

3. Predict on a sample text:

```bash
python models/predict.py "Free prize, click here"
```

OpenSpec artifacts live under `openspec/changes/` and `openspec/project.md` documents project conventions.

Run the Streamlit demo locally:

```bash
# start streamlit from the repository root
streamlit run app/streamlit_app.py
```

