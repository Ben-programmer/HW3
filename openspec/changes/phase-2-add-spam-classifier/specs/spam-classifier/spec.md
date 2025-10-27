++ mkdir mode 0755
## ADDED Requirements

### Requirement: Baseline Spam Classifier
The system SHALL provide a baseline spam classifier trained using Logistic Regression on the provided SMS/Email dataset. The baseline SHALL include data ingestion, preprocessing, TF-IDF feature extraction, training, and evaluation steps.

#### Scenario: Training completes and outputs metrics
- **GIVEN** the dataset has been downloaded and validated
- **WHEN** the training script is executed for the baseline model
- **THEN** the training run SHALL complete without error and produce a report including precision, recall, F1-score, and a confusion matrix

#### Scenario: Prediction returns label and probability
- **GIVEN** a trained baseline model artifact is available
- **WHEN** a message text is passed to the prediction script
- **THEN** the script SHALL return a prediction object containing: `label` (spam/ham) and `probability` (0.0-1.0)

### Requirement: Data source referenced
The spec SHALL reference the canonical data source location used for the baseline and note required pre-processing (e.g., header handling, label column mapping).

#### Scenario: Data source accessible
- **GIVEN** the data source URL is reachable
- **WHEN** the download script runs
- **THEN** the dataset SHALL be saved in `data/raw/` and include the expected columns for label and message (or a conversion script shall run)
