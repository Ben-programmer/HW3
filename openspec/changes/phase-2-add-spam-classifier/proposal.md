++ mkdir mode 0755
## Why
垃圾郵件（spam）分類是一個常見且有用的功能，可用來自動篩選短訊或電子郵件以減少使用者噪音。建立一個以機器學習為基礎的 spam-classifier 能作為後續改進（例如更好的特徵工程、模型蒐集與監控）的基礎工作。

## What Changes
- 新增 capability `spam-classifier`，並在 `changes/add-spam-classifier/` 下提供 proposal、tasks 與 spec delta。
- Phase1（baseline）：使用 Logistic Regression 作為 baseline 模型，資料來源為公開資料集（see Data Source）。實作資料下載、前處理、特徵工程（TF-IDF）、模型訓練與簡單評估報告。
- PhaseN：保留多個後續階段項目，目前留空以便日後填入（例如 hyperparameter tuning、model comparison、deployment、monitoring）。

## Impact
- Affected specs: 新增 `spam-classifier` 能力（spec）
- Affected code: 新增 Python 訓練/預測腳本（建議路徑 `models/spam_classifier/`）與資料下載腳本
- Risk: 中等。資料需先檢查格式與質量，訓練結果可能需反覆調整。

## Data Source
- CSV: https://github.com/huanchen1107/2025ML-spamEmail/blob/main/datasets/sms_spam_no_header.csv
- 假設：CSV 包含 `label`、`message` 或類似欄位；如欄位不同需先確認與轉換。

## Acceptance Criteria
- 可執行的資料下載與前處理腳本。
- 使用 Logistic Regression 訓練並輸出基本評估（precision, recall, F1, 混淆矩陣）。
- `openspec/changes/add-spam-classifier/specs/spam-classifier/spec.md` 包含至少一個 ADDED Requirement 與 Scenario，描述訓練成功與模型輸出行為。

## Next Steps
1. 實作 Phase1 任務並在 tasks.md 中逐一完成。
2. 在 PR 中請 reviewer 檢閱 proposal，通過後執行實作並最後 archive change。
