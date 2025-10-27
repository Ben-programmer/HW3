++ mkdir mode 0755
## 1. Phase1 - Baseline (Logistic Regression)
- [ ] 1.1 Download dataset from the provided URL and inspect schema (`data/download_dataset.py`)
- [ ] 1.2 Implement data sanitation and preprocessing (`data/preprocess.py`) — lowercasing, punctuation removal, optional stopword removal
- [ ] 1.3 Implement feature extraction using TF-IDF (`features/tfidf.py`)
- [ ] 1.4 Implement training script using scikit-learn LogisticRegression (`models/train_baseline.py`)
- [ ] 1.5 Evaluate model and produce report (precision, recall, F1, confusion matrix) (`reports/evaluate.md` or printed output)
- [ ] 1.6 Add a minimal prediction script (`models/predict.py`) that loads model and returns label + probability
- [ ] 1.7 Add unit tests for preprocessing and a small integration test to train on a tiny sample

## 2. Phase2..N (placeholders)
- [ ] 2.1 (empty)
- [ ] 2.2 (empty)

## 3. Validation
- [ ] 3.1 Run `openspec validate add-spam-classifier --strict` and fix any spec formatting issues
