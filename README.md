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

Run the Streamlit demo locally:

```bash
# start streamlit from the repository root
streamlit run app/streamlit_app.py
```

Streamlit demo - UI & parameters

Start the Streamlit UI and open http://localhost:8501. The left sidebar contains dataset and training controls; the right side has pages (Overview, Predict, Evaluation, Artifacts).

Sidebar controls (training):

- Choose dataset: select a CSV from the `data/` folder or upload a new CSV (saved as `data/uploaded_dataset.csv`).
- Label column: column name in CSV for labels (default `label`). If CSV has no header the app will auto-detect and fall back to headerless parsing.
- Text column: column name in CSV for text (default `message`).
- Models dir: output directory for artifacts (default `models/artifacts`).
- Text Size: TF-IDF `max_features` (default 10000).
- Seed: random seed for train/test split (default 42).
- Decision threshold: probability threshold to map predicted probabilities to binary labels.

Profiles

You can save parameter profiles in the sidebar and quickly load them later (saved under `profiles/` in the repo).

Pages:

- Overview: brief description and current model status.
- Predict: enter a message and get label + probability from the latest model.
- Evaluation: shows training metrics (precision/recall/F1), confusion matrix, and the training parameters that produced the model.
- Artifacts: list and download model/vectorizer files.

All training runs write a `report.json` to the artifacts directory which the UI reads to display metrics and parameters.

Development notes

- `models/train_baseline.py` supports extra CLI flags: `--label_col`, `--text_col`, `--max_features`, `--seed`, `--threshold`.
- `app/streamlit_app.py` provides parameters in the sidebar and background training.
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

