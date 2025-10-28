import streamlit as st
from pathlib import Path
import subprocess
import json
from collections import Counter
import sys

# Ensure project root is importable when Streamlit runs this file as a script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.predict import predict
from data.preprocess import load_dataset, preprocess_df, clean_text, classify_token


ARTIFACT_MODEL = Path("models/artifacts/logreg_baseline.joblib")
ARTIFACT_VEC = Path("models/artifacts/tfidf_vectorizer.joblib")


def show_model_status():
    if ARTIFACT_MODEL.exists() and ARTIFACT_VEC.exists():
        st.success("Model artifacts found.")
        st.write(f"Model: {ARTIFACT_MODEL}")
        st.write(f"Vectorizer: {ARTIFACT_VEC}")
    else:
        st.warning("No model artifacts found. Please run training first (see Train in sidebar).")


def _read_report(out_dir: str = "models/artifacts"):
    import json
    from pathlib import Path

    p = Path(out_dir) / "report.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def run_training():
    st.info("Starting training in background thread...")

    def _worker(
        data_path: str = "data/raw/sms_spam_no_header.csv",
        out: str = "models/artifacts",
        label_col: str | None = None,
        text_col: str | None = None,
        max_features: int = 10000,
        seed: int = 42,
        threshold: float = 0.5,
    ):
        # run training and write report (train_baseline already writes report.json)
        try:
            from models.train_baseline import train

            report = train(
                data_path,
                out,
                label_col=label_col,
                text_col=text_col,
                max_features=max_features,
                seed=seed,
                threshold=threshold,
            )
            # write a simple log file
            Path(out).mkdir(parents=True, exist_ok=True)
            (Path(out) / "train.log").write_text(str(report))
        except Exception as e:
            Path(out).mkdir(parents=True, exist_ok=True)
            (Path(out) / "train.log").write_text(f"ERROR: {e}")

    data_path = "data/raw/sms_spam_no_header.csv"
    # prefer uploaded file if present in session state
    if st.session_state.get("uploaded_dataset_path"):
        data_path = st.session_state["uploaded_dataset_path"]

    import threading

    out_dir = st.session_state.get("models_dir", "models/artifacts")
    label_col = st.session_state.get("label_col")
    text_col = st.session_state.get("text_col")
    max_features = st.session_state.get("max_features", 10000)
    seed = st.session_state.get("seed", 42)
    threshold = st.session_state.get("threshold", 0.5)

    th = threading.Thread(
        target=_worker,
        args=(data_path, out_dir, label_col, text_col, max_features, seed, threshold),
        daemon=True,
    )
    st.session_state["training_thread"] = th
    st.session_state["training_started_at"] = True
    th.start()
    st.success("Training started in background. Refresh status after a moment.")


def main():
    st.set_page_config(page_title="Spam Classifier", layout="wide")

    # Sidebar (left menu)
    with st.sidebar:
        st.title("Spam Classifier")
        st.markdown("A small demo to train and evaluate a spam classifier.")

        # dataset selector: list CSVs in `data/` and allow uploading a new one
        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)
        csv_files = sorted([p.name for p in data_dir.glob("*.csv")])
        options = ["-- upload new file --"] + csv_files
        # default directly to data/sms_spam_no_header.csv when present
        if "sms_spam_no_header.csv" in options:
            default_index = options.index("sms_spam_no_header.csv")
        else:
            default_index = 0
        choice = st.selectbox("Choose dataset", options, index=default_index)

        if choice == "-- upload new file --":
            uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
            if uploaded is not None:
                upload_path = data_dir / "uploaded_dataset.csv"
                with open(upload_path, "wb") as fh:
                    fh.write(uploaded.getbuffer())
                st.success(f"Uploaded dataset saved to {upload_path}")
                st.session_state["uploaded_dataset_path"] = str(upload_path)
        else:
            selected_path = data_dir / choice
            st.info(f"Selected dataset: {selected_path}")
            st.session_state["uploaded_dataset_path"] = str(selected_path)

        st.markdown("---")

        # training & parameters
        st.subheader("Training parameters")
        # bind widgets to session state keys so loading profiles can update them
        lbl_col = st.text_input("Label column", value=st.session_state.get("label_col", "label"), key="label_col_input")
        txt_col = st.text_input("Text column", value=st.session_state.get("text_col", "message"), key="text_col_input")
        models_dir = st.text_input("Models dir", value=st.session_state.get("models_dir", "models/artifacts"), key="models_dir_input")
        max_feats = st.number_input("Text Size (max features)", min_value=100, max_value=50000, value=st.session_state.get("max_features", 10000), step=100, key="max_feats_input")
        seed = st.number_input("Seed", min_value=0, value=st.session_state.get("seed", 42), step=1, key="seed_input")
        threshold = st.slider("Decision threshold", min_value=0.0, max_value=1.0, value=st.session_state.get("threshold", 0.5), key="threshold_input")

        # persist into session state so run_training can pick them up
        st.session_state["label_col"] = lbl_col
        st.session_state["text_col"] = txt_col
        st.session_state["models_dir"] = models_dir
        st.session_state["max_features"] = int(max_feats)
        st.session_state["seed"] = int(seed)
        st.session_state["threshold"] = float(threshold)

        if st.button("Train baseline model"):
            run_training()

        if st.button("Download dataset"):
            try:
                subprocess.run([sys.executable, "data/download_dataset.py"], check=True)
                st.success("Dataset downloaded to data/raw/")
            except Exception as e:
                st.error(f"Download failed: {e}")

        st.markdown("---")
        st.subheader("Quick status")
        report = _read_report("models/artifacts")
        if report:
            st.metric("F1", f"{report.get('f1'):.3f}")
            st.metric("Recall", f"{report.get('recall'):.3f}")
            st.metric("Precision", f"{report.get('precision'):.3f}")
        else:
            st.info("No trained model yet")

    # Main area (right)
    page = st.radio("Page", ["Overview", "Predict", "Evaluation", "Artifacts"], horizontal=True)

    if page == "Overview":
        st.header("Overview")
        st.write("This demo shows a Logistic Regression baseline for SMS spam classification. Use the left menu to choose a dataset, start training, and download artifacts.")
        st.subheader("Model status")
        show_model_status()
        # Data overview: class distribution and token replacements
        st.subheader("Data Overview")
        # small CSS for card-style sections and font tweaks
        st.markdown(
            """
            <style>
            .card { background: #fbfdff; padding: 14px; border-radius:10px; box-shadow: 0 4px 16px rgba(11,95,255,0.06); }
            .section-title { font-family: 'Helvetica Neue', Arial, sans-serif; color:#0b5fff; font-weight:600; }
            .small-muted { color: #6c757d; font-size:0.9rem }
            </style>
            """,
            unsafe_allow_html=True,
        )

        data_path = st.session_state.get("uploaded_dataset_path", "data/sms_spam_no_header.csv")
        try:
            df_raw = load_dataset(data_path)
            df_clean = preprocess_df(df_raw)

            # class distribution (card)
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Class distribution</div>", unsafe_allow_html=True)
            counts = df_clean["label"].value_counts()
            st.bar_chart(counts)
            st.markdown("</div>", unsafe_allow_html=True)

            # token replacements: map original token -> cleaned token where changed
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Token replacements in cleaned text (top 20)</div>", unsafe_allow_html=True)
            mapping = Counter()
            for orig_msg in df_raw["message"].astype(str).tolist():
                for token in orig_msg.split():
                    cleaned = clean_text(token)
                    if cleaned and cleaned != token.lower():
                        cat = classify_token(token)
                        mapping[(token, cleaned, cat)] += 1

            if mapping:
                rows = [ {"original": k[0], "cleaned": k[1], "category": k[2], "count": v} for k, v in mapping.most_common(20) ]
                st.table(rows)
            else:
                st.info("No token replacements detected or dataset empty.")

            st.markdown("</div>", unsafe_allow_html=True)
            # Top tokens by class (ham vs spam)
            try:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>Top tokens by class</div>", unsafe_allow_html=True)
                top_n = st.number_input("Top N tokens", min_value=5, max_value=50, value=10, key="top_n_tokens")

                from collections import Counter as _Counter
                import matplotlib.pyplot as plt
                import numpy as _np

                cnt_ham = _Counter()
                cnt_spam = _Counter()
                # df_clean messages are already cleaned (lowercased, punctuation removed), split on whitespace
                for lbl, msg in df_clean[["label", "message"]].values:
                    for tok in str(msg).split():
                        if lbl == "ham":
                            cnt_ham[tok] += 1
                        else:
                            cnt_spam[tok] += 1

                ham_top = cnt_ham.most_common(int(top_n))
                spam_top = cnt_spam.most_common(int(top_n))

                # prepare side-by-side charts using Streamlit native plotting for cloud compatibility
                if ham_top or spam_top:
                    col1, col2 = st.columns(2)
                    if ham_top:
                        import pandas as _pd
                        ham_df = _pd.DataFrame(ham_top, columns=["token", "count"]).set_index("token")
                        with col1:
                            st.markdown("**Top tokens — ham**")
                            st.bar_chart(ham_df)
                    else:
                        with col1:
                            st.info("No ham tokens")

                    if spam_top:
                        import pandas as _pd
                        spam_df = _pd.DataFrame(spam_top, columns=["token", "count"]).set_index("token")
                        with col2:
                            st.markdown("**Top tokens — spam**")
                            st.bar_chart(spam_df)
                    else:
                        with col2:
                            st.info("No spam tokens")
                else:
                    st.info("No tokens found to display top tokens by class.")

                st.markdown("</div>", unsafe_allow_html=True)
            except Exception:
                # non-fatal; do not break overview if plotting fails
                pass
        except Exception as e:
            st.warning(f"Could not load dataset for overview: {e}")

    elif page == "Predict":
        st.header("Try a sample message")
        text = st.text_area("Message text", "Free entry: you won a prize! Click to claim.")
        if st.button("Predict"):
            try:
                res = predict(text)
                st.json(res)
                if res.get("probability") is not None:
                    st.progress(min(max(res["probability"], 0.0), 1.0))
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    elif page == "Evaluation":
        st.header("Training status & report")
        if st.button("Refresh status"):
            try:
                st.experimental_rerun()
            except Exception:
                st.info("Automatic refresh not available. Please reload the page manually.")

        if st.session_state.get("training_started_at") and st.session_state.get("training_thread"):
            th = st.session_state["training_thread"]
            if th.is_alive():
                st.info("Training is still running...")
            else:
                st.success("Background training thread finished.")

        report = _read_report("models/artifacts")
        if report:
            st.subheader("Metrics")
            # wrap metrics in a card for visual emphasis
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Precision", f"{report.get('precision'):.3f}")
            c2.metric("Recall", f"{report.get('recall'):.3f}")
            c3.metric("F1", f"{report.get('f1'):.3f}")
            st.markdown("</div>", unsafe_allow_html=True)
            cm = report.get("confusion_matrix")
            if cm:
                try:
                    # display confusion matrix as a simple table for cloud compatibility
                    import pandas as _pd
                    cm_arr = cm
                    cm_df = _pd.DataFrame(cm_arr, index=["actual_ham", "actual_spam"], columns=["pred_ham", "pred_spam"])
                    st.markdown("**Confusion matrix**")
                    st.table(cm_df)
                except Exception:
                    st.write(cm)

            model_path = report.get("model_path")
            if model_path:
                try:
                    with open(model_path, "rb") as fh:
                        model_bytes = fh.read()
                    st.download_button("Download model artifact", model_bytes, file_name=Path(model_path).name)
                except Exception as e:
                    st.warning(f"Could not open model file for download: {e}")
            # display training parameters from report
            st.subheader("Training parameters used")
            params = {k: report.get(k) for k in ["threshold", "max_features", "seed"] if k in report}
            if params:
                st.table(params.items())

            # Model performance (threshold sweep) on current dataset
            try:
                st.subheader("Model performance (threshold sweep)")
                from pathlib import Path as _Path
                from joblib import load as _load
                import numpy as _np
                from sklearn.metrics import precision_recall_fscore_support
                import pandas as _pd

                vec_path = report.get("vectorizer_path") or "models/artifacts/tfidf_vectorizer.joblib"
                model_path = report.get("model_path") or "models/artifacts/logreg_baseline.joblib"
                p_vec = _Path(vec_path)
                p_model = _Path(model_path)
                if p_vec.exists() and p_model.exists():
                    vec = _load(p_vec)
                    clf = _load(p_model)

                    # load dataset for evaluation (use cleaned messages)
                    eval_path = st.session_state.get("uploaded_dataset_path", "data/sms_spam_no_header.csv")
                    df_eval = preprocess_df(load_dataset(eval_path))
                    X = vec.transform(df_eval["message"].astype(str).tolist())
                    y = (df_eval["label"] == "spam").astype(int).values

                    thresholds = list(_np.linspace(0.0, 1.0, 11))
                    rows = []
                    probs = None
                    try:
                        probs = clf.predict_proba(X)[:, 1]
                    except Exception:
                        # classifier may not support predict_proba
                        try:
                            probs = clf.decision_function(X)
                            # scale to 0..1
                            probs = (_np.tanh(probs) + 1) / 2
                        except Exception:
                            probs = None

                    if probs is None:
                        st.info("Model does not provide probabilities; threshold sweep unavailable.")
                    else:
                        for t in thresholds:
                            preds = (probs >= t).astype(int)
                            try:
                                p, r, f, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
                            except Exception:
                                p = r = f = 0.0
                            rows.append({"threshold": round(float(t), 2), "precision": round(float(p), 3), "recall": round(float(r), 3), "f1": round(float(f), 3)})

                        df_metrics = _pd.DataFrame(rows)
                        st.table(df_metrics)
                else:
                    st.info("Model artifacts not found for threshold sweep.")
            except Exception as e:
                st.warning(f"Could not run threshold sweep: {e}")
        else:
            st.info("No report found. Run training to produce metrics.")

    elif page == "Artifacts":
        st.header("Artifacts")
        art_dir = Path("models/artifacts")
        if art_dir.exists():
            files = sorted([p.name for p in art_dir.iterdir()])
            for fname in files:
                p = art_dir / fname
                st.write(fname)
                if p.suffix in [".joblib", ".pkl"]:
                    try:
                        with open(p, "rb") as fh:
                            st.download_button(f"Download {fname}", fh.read(), file_name=fname)
                    except Exception:
                        st.write("(cannot download)")
        else:
            st.info("No artifacts directory found. Run training first.")


if __name__ == "__main__":
    main()
