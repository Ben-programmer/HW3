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

        # parameter profiles
        profiles_dir = Path("profiles")
        profiles_dir.mkdir(parents=True, exist_ok=True)
        profile_files = sorted([p.name for p in profiles_dir.glob("*.json")])
        profile_choice = st.selectbox("Profile", ["-- new --"] + profile_files, index=0)
        profile_name = st.text_input("Profile name (for save)", value="", key="profile_name")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save profile"):
                name = profile_name.strip() or "default"
                ppath = profiles_dir / f"{name}.json"
                payload = {
                    "label_col": lbl_col,
                    "text_col": txt_col,
                    "models_dir": models_dir,
                    "max_features": int(max_feats),
                    "seed": int(seed),
                    "threshold": float(threshold),
                }
                ppath.write_text(json.dumps(payload, indent=2))
                st.success(f"Saved profile {ppath}")
        with col2:
            if st.button("Load profile"):
                if profile_choice != "-- new --":
                    pdata = json.loads((profiles_dir / profile_choice).read_text())
                    # update session state keys and rerun to refresh widgets
                    st.session_state["label_col"] = pdata.get("label_col")
                    st.session_state["text_col"] = pdata.get("text_col")
                    st.session_state["models_dir"] = pdata.get("models_dir")
                    st.session_state["max_features"] = int(pdata.get("max_features", 10000))
                    st.session_state["seed"] = int(pdata.get("seed", 42))
                    st.session_state["threshold"] = float(pdata.get("threshold", 0.5))
                    try:
                        st.experimental_rerun()
                    except Exception:
                        st.info("Profile loaded; please refresh the page to see updated values.")

        # persist into session state so run_training can pick them up
        st.session_state["label_col"] = st.session_state.get("label_col", lbl_col)
        st.session_state["text_col"] = st.session_state.get("text_col", txt_col)
        st.session_state["models_dir"] = st.session_state.get("models_dir", models_dir)
        st.session_state["max_features"] = int(st.session_state.get("max_features", max_feats))
        st.session_state["seed"] = int(st.session_state.get("seed", seed))
        st.session_state["threshold"] = float(st.session_state.get("threshold", threshold))

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
            # more prominent metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Precision", f"{report.get('precision'):.3f}")
            c2.metric("Recall", f"{report.get('recall'):.3f}")
            c3.metric("F1", f"{report.get('f1'):.3f}")
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
