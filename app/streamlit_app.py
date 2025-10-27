import streamlit as st
from pathlib import Path
import subprocess
import json
import sys

# Ensure project root is importable when Streamlit runs this file as a script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.predict import predict


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

    def _worker(data_path: str = "data/raw/sms_spam_no_header.csv", out: str = "models/artifacts"):
        # run training and write report (train_baseline already writes report.json)
        try:
            from models.train_baseline import train

            report = train(data_path, out)
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

    th = threading.Thread(target=_worker, args=(data_path, "models/artifacts"), daemon=True)
    st.session_state["training_thread"] = th
    st.session_state["training_started_at"] = True
    th.start()
    st.success("Training started in background. Refresh status after a moment.")


def main():
    st.set_page_config(page_title="Spam Classifier", layout="centered")
    st.title("Spam Classifier — Demo")

    with st.sidebar:
        st.header("Actions")
        # dataset selector: list CSVs in `data/` and allow uploading a new one
        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)
        csv_files = sorted([p.name for p in data_dir.glob("*.csv")])
        options = ["-- upload new file --"] + csv_files
        # default to sms_spam_no_header.csv if present
        default_index = 1 if "sms_spam_no_header.csv" in csv_files else 0
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
            # use selected file from data/ as dataset
            selected_path = data_dir / choice
            st.info(f"Selected dataset: {selected_path}")
            st.session_state["uploaded_dataset_path"] = str(selected_path)

        if st.button("Train baseline model"):
            run_training()
        st.markdown("---")
        st.markdown("Data & Model")
        if st.button("Download dataset"):
            try:
                # use the running python executable to avoid 'python' not found
                subprocess.run([sys.executable, "data/download_dataset.py"], check=True)
                st.success("Dataset downloaded to data/raw/")
            except Exception as e:
                st.error(f"Download failed: {e}")

    st.header("Model status")
    show_model_status()

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

    # status / report
    st.header("Training status & report")
    if st.button("Refresh status"):
        # experimental_rerun may be missing in some Streamlit builds/environments.
        try:
            st.experimental_rerun()
        except Exception:
            # Fallback: inform user to manually reload if automatic rerun isn't available.
            st.info("Automatic refresh not available in this Streamlit build. Please reload the page manually.")

    if st.session_state.get("training_started_at") and st.session_state.get("training_thread"):
        th = st.session_state["training_thread"]
        if th.is_alive():
            st.info("Training is still running...")
        else:
            st.success("Background training thread finished.")

    report = _read_report("models/artifacts")
    if report:
        st.subheader("Metrics")
        st.write(f"Precision: {report.get('precision')}")
        st.write(f"Recall: {report.get('recall')}")
        st.write(f"F1: {report.get('f1')}")
        cm = report.get("confusion_matrix")
        if cm:
            try:
                import matplotlib.pyplot as plt
                import numpy as np

                fig, ax = plt.subplots()
                im = ax.imshow(np.array(cm), cmap="Blues")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                for (i, j), z in np.ndenumerate(np.array(cm)):
                    ax.text(j, i, str(z), ha="center", va="center")
                st.pyplot(fig)
            except Exception:
                st.write(cm)

        # offer to download the actual model file (binary) when available
        model_path = report.get("model_path")
        if model_path:
            try:
                with open(model_path, "rb") as fh:
                    model_bytes = fh.read()
                st.download_button("Download model artifact", model_bytes, file_name=Path(model_path).name)
            except Exception as e:
                st.warning(f"Could not open model file for download: {e}")

    st.header("About")
    st.markdown(
        "This demo uses a Logistic Regression baseline. Use the sidebar to download the dataset and train the baseline."
    )


if __name__ == "__main__":
    main()
