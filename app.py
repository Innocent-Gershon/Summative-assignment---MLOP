"""
app.py - Streamlit UI for Weather Image Classifier
Active learning flow: predict an image, correct the label, retrain immediately.
Run: streamlit run app.py
"""

import io
import requests
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

API_URL = "http://localhost:8000"
CLASSES = ["Cloudy", "Rain", "Shine", "Sunrise"]

st.set_page_config(page_title="Weather Classifier", layout="wide")


# -- Helpers -------------------------------------------------------------------
def api_get(path):
    try:
        r = requests.get(API_URL + path, timeout=10)
        return r.json(), r.status_code
    except Exception as e:
        return {"error": str(e)}, 0


def api_post(path, **kwargs):
    try:
        r = requests.post(API_URL + path, timeout=120, **kwargs)
        return r.json(), r.status_code
    except Exception as e:
        return {"error": str(e)}, 0


# -- Session state defaults ----------------------------------------------------
if "predict_image_bytes" not in st.session_state:
    st.session_state["predict_image_bytes"] = None
if "predict_image_name" not in st.session_state:
    st.session_state["predict_image_name"] = None
if "predict_result" not in st.session_state:
    st.session_state["predict_result"] = None


# -- Sidebar -------------------------------------------------------------------
st.sidebar.title("Weather Classifier")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["Status", "Predict", "Visualizations", "Upload Data", "Retrain"])

status_data, _ = api_get("/status")
if "uptime_seconds" in status_data:
    st.sidebar.metric("Uptime (s)", status_data["uptime_seconds"])
    st.sidebar.metric("Total Requests", status_data["total_requests"])
    st.sidebar.metric("Avg Latency (ms)", status_data["avg_latency_ms"])
    model_ok = status_data.get("model_loaded", False)
    st.sidebar.markdown(f"**Model:** {'Loaded' if model_ok else 'Not trained'}")
else:
    st.sidebar.warning("API not reachable at " + API_URL)

st.sidebar.markdown("---")
st.sidebar.caption("FastAPI backend: " + API_URL)

# ==============================================================================
# PAGE: STATUS
# ==============================================================================
if page == "Status":
    st.title("Model Status and Uptime")

    if "error" in status_data:
        st.error(f"Cannot reach API: {status_data['error']}")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Uptime (s)", status_data.get("uptime_seconds", "--"))
        col2.metric("Total Requests", status_data.get("total_requests", "--"))
        col3.metric("Avg Latency (ms)", status_data.get("avg_latency_ms", "--"))

        st.markdown("---")
        model_loaded = status_data.get("model_loaded", False)
        st.markdown(f"**Model status:** {'Loaded' if model_loaded else 'Not trained yet'}")
        st.markdown(f"**Model path:** `{status_data.get('model_path', '--')}`")
        st.markdown(f"**Classes:** {', '.join(status_data.get('classes', []))}")

        training = status_data.get("training", {})
        st.markdown("---")
        st.subheader("Training State")
        if training.get("running"):
            st.info(f"Training in progress — type: **{training['type']}**, started: {training['started_at']}")
        elif training.get("error"):
            st.error("Last training run encountered an error.")
            st.code(training["error"])
        elif training.get("finished_at"):
            st.success(f"Last training finished at {training['finished_at']}")
        else:
            st.write("Idle — no training has run yet.")

    if st.button("Refresh"):
        st.rerun()

# ==============================================================================
# PAGE: PREDICT
# ==============================================================================
elif page == "Predict":
    st.title("Single Image Prediction")

    uploaded = st.file_uploader("Upload a weather image", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img, caption=uploaded.name, use_column_width=True)

        with col2:
            if st.button("Predict", type="primary"):
                with st.spinner("Running inference..."):
                    uploaded.seek(0)
                    img_bytes = uploaded.read()
                    result, code = api_post(
                        "/predict",
                        files={"file": (uploaded.name, img_bytes, uploaded.type)},
                    )
                if code == 200:
                    # Store image and result in session state for Retrain page
                    st.session_state["predict_image_bytes"] = img_bytes
                    st.session_state["predict_image_name"] = uploaded.name
                    st.session_state["predict_result"] = result

                    st.success(f"**Predicted class:** {result['predicted_class']}")
                    st.metric("Confidence", f"{result['confidence']*100:.1f}%")
                    st.markdown("**Class probabilities:**")
                    probs = result["probabilities"]
                    classes = list(probs.keys())
                    values = [probs[c] for c in classes]
                    fig, ax = plt.subplots(figsize=(5, 2.5))
                    colors = ["#38bdf8" if c == result["predicted_class"] else "#475569" for c in classes]
                    bars = ax.barh(classes, values, color=colors)
                    ax.set_xlim(0, 1)
                    ax.set_xlabel("Probability")
                    ax.bar_label(bars, fmt="%.2f", padding=3)
                    ax.set_facecolor("#0f172a")
                    fig.patch.set_facecolor("#1e293b")
                    ax.tick_params(colors="white")
                    ax.xaxis.label.set_color("white")
                    st.pyplot(fig)

                    st.info("If the prediction is wrong, go to the **Retrain** page — the image will be waiting for you there.")
                elif code == 503:
                    st.error("Model not trained yet. Go to Retrain first.")
                else:
                    st.error(f"Error {code}: {result.get('detail', result)}")

# ==============================================================================
# PAGE: VISUALIZATIONS
# ==============================================================================
elif page == "Visualizations":
    st.title("Data and Model Visualizations")

    st.subheader("Feature 1 - Class Distribution")
    dist_data, _ = api_get("/visualizations/class-distribution")
    if "train" in dist_data:
        classes = list(dist_data["train"].keys())
        train_vals = [dist_data["train"][c] for c in classes]
        test_vals = [dist_data["test"][c] for c in classes]
        x = np.arange(len(classes))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x - 0.2, train_vals, 0.4, label="Train", color="#38bdf8")
        ax.bar(x + 0.2, test_vals,  0.4, label="Test",  color="#4ade80")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, color="white")
        ax.set_ylabel("Image Count", color="white")
        ax.set_title("Images per Class", color="white")
        ax.legend()
        ax.set_facecolor("#0f172a")
        fig.patch.set_facecolor("#1e293b")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        st.pyplot(fig)
        st.info(dist_data.get("interpretation", ""))
    else:
        st.warning("Could not load class distribution data.")

    st.markdown("---")

    st.subheader("Feature 2 - Sample Images per Class")
    from pathlib import Path
    TEST_DIR = Path(__file__).parent / "data" / "test"
    cols = st.columns(4)
    found_any = False
    for i, cls in enumerate(CLASSES):
        cls_dir = TEST_DIR / cls
        images = []
        if cls_dir.exists():
            images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpeg"))
        if images:
            found_any = True
            img = Image.open(images[0]).convert("RGB")
            cols[i].image(img, caption=cls, use_column_width=True)
        else:
            cols[i].markdown(f"**{cls}**\n\n*(no images)*")
    if found_any:
        st.info(
            "Each weather class has distinct visual characteristics. "
            "Sunrise images show warm orange/pink hues. Shine images are bright with blue sky. "
            "Cloudy images are grey and overcast. Rain images show dark skies and precipitation."
        )

    st.markdown("---")

    st.subheader("Feature 3 - Average Model Confidence per Class")
    conf_data, _ = api_get("/visualizations/model-confidence")
    if "avg_confidence_per_class" in conf_data:
        conf = conf_data["avg_confidence_per_class"]
        classes = list(conf.keys())
        values = [conf[c] for c in classes]
        fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(classes), endpoint=False).tolist()
        values_plot = values + [values[0]]
        angles += angles[:1]
        ax.plot(angles, values_plot, color="#38bdf8", linewidth=2)
        ax.fill(angles, values_plot, color="#38bdf8", alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(classes, color="white", size=11)
        ax.set_ylim(0, 1)
        ax.set_facecolor("#0f172a")
        fig.patch.set_facecolor("#1e293b")
        ax.tick_params(colors="white")
        ax.yaxis.set_tick_params(labelcolor="white")
        st.pyplot(fig)
        st.info(conf_data.get("interpretation", ""))
    else:
        st.warning(conf_data.get("message", "Model not trained yet."))

# ==============================================================================
# PAGE: UPLOAD DATA
# ==============================================================================
elif page == "Upload Data":
    st.title("Upload New Training Data")
    st.markdown("Browse and select images from your local machine to add to the training set.")

    label = st.selectbox("Weather class label", CLASSES)
    files = st.file_uploader(
        "Select images from your computer (multiple allowed)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if files:
        st.write(f"**{len(files)} file(s) selected**")
        preview_cols = st.columns(min(len(files), 5))
        for i, f in enumerate(files[:5]):
            f.seek(0)
            preview_cols[i].image(Image.open(f).convert("RGB"), caption=f.name, use_column_width=True)

    if st.button("Upload Images", type="primary", disabled=not files):
        multipart = []
        for f in files:
            f.seek(0)
            multipart.append(("files", (f.name, f.read(), f.type)))
        with st.spinner(f"Uploading {len(files)} image(s) as '{label}'..."):
            result, code = api_post(f"/upload?label={label}", files=multipart)
        if code == 200:
            st.success(f"Uploaded **{result['uploaded']}** image(s) as **{result['label']}**.")
            st.balloons()
        else:
            st.error(f"Upload failed: {result.get('detail', result)}")

# ==============================================================================
# PAGE: RETRAIN
# ==============================================================================
elif page == "Retrain":
    st.title("Retrain Model")

    # Check if an image came from the Predict page
    from_predict = st.session_state.get("predict_image_bytes") is not None
    predict_result = st.session_state.get("predict_result")

    if from_predict:
        st.info(
            f"Image carried over from Predict page. "
            f"Model predicted: **{predict_result['predicted_class']}** "
            f"({predict_result['confidence']*100:.1f}% confidence). "
            f"Select the correct label below and retrain."
        )
        img = Image.open(io.BytesIO(st.session_state["predict_image_bytes"])).convert("RGB")

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img, caption=st.session_state["predict_image_name"], use_column_width=True)

        with col2:
            # Default label to predicted class; user can correct it
            predicted_class = predict_result["predicted_class"]
            default_idx = CLASSES.index(predicted_class) if predicted_class in CLASSES else 0
            correct_label = st.selectbox(
                "Correct label for this image",
                CLASSES,
                index=default_idx,
                help="If the model was wrong, change this to the correct class before retraining.",
            )
            retrain_epochs = st.number_input("Epochs", min_value=1, max_value=50, value=10)

            if st.button("Trigger Retraining", type="primary"):
                # Upload the image with the correct label
                img_bytes = st.session_state["predict_image_bytes"]
                img_name = st.session_state["predict_image_name"]
                with st.spinner(f"Uploading image as '{correct_label}'..."):
                    upload_result, upload_code = api_post(
                        f"/upload?label={correct_label}",
                        files=[("files", (img_name, img_bytes, "image/jpeg"))],
                    )
                if upload_code != 200:
                    st.error(f"Upload failed: {upload_result.get('detail', upload_result)}")
                else:
                    st.success(f"Image uploaded as **{correct_label}**.")
                    with st.spinner("Starting retraining..."):
                        result, code = api_post(f"/retrain?epochs={retrain_epochs}")
                    if code == 200:
                        st.success(result["message"] + f" — {result.get('total_images', '?')} images found.")
                        # Clear session state after successful retrain
                        st.session_state["predict_image_bytes"] = None
                        st.session_state["predict_image_name"] = None
                        st.session_state["predict_result"] = None
                    elif code == 409:
                        st.warning(result["detail"])
                    else:
                        st.error(str(result))

        if st.button("Clear and upload different images instead"):
            st.session_state["predict_image_bytes"] = None
            st.session_state["predict_image_name"] = None
            st.session_state["predict_result"] = None
            st.rerun()

    else:
        # No image from Predict — manual upload flow
        st.markdown("Upload images from your computer, select the correct label, and retrain.")
        label = st.selectbox("Weather class label", CLASSES)
        files = st.file_uploader(
            "Select images from your computer (multiple allowed)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

        if files:
            st.write(f"**{len(files)} file(s) selected**")
            preview_cols = st.columns(min(len(files), 5))
            for i, f in enumerate(files[:5]):
                f.seek(0)
                preview_cols[i].image(Image.open(f).convert("RGB"), caption=f.name, use_column_width=True)

            retrain_epochs = st.number_input("Epochs", min_value=1, max_value=50, value=10)

            if st.button("Trigger Retraining", type="primary"):
                multipart = []
                for f in files:
                    f.seek(0)
                    multipart.append(("files", (f.name, f.read(), f.type)))
                with st.spinner(f"Uploading {len(files)} image(s) as '{label}'..."):
                    upload_result, upload_code = api_post(f"/upload?label={label}", files=multipart)
                if upload_code != 200:
                    st.error(f"Upload failed: {upload_result.get('detail', upload_result)}")
                else:
                    st.success(f"Uploaded **{upload_result['uploaded']}** image(s) as **{upload_result['label']}**.")
                    with st.spinner("Starting retraining..."):
                        result, code = api_post(f"/retrain?epochs={retrain_epochs}")
                    if code == 200:
                        st.success(result["message"] + f" — {result.get('total_images', '?')} images found.")
                    elif code == 409:
                        st.warning(result["detail"])
                    else:
                        st.error(str(result))

    st.markdown("---")
    st.subheader("Training Progress")
    train_status, _ = api_get("/training-status")
    if train_status.get("running"):
        st.info(f"**{train_status['type'].capitalize()}** in progress since {train_status['started_at']}")
        if st.button("Refresh"):
            st.rerun()
    elif train_status.get("error"):
        st.error("Training error occurred.")
        st.code(train_status["error"])
    elif train_status.get("finished_at"):
        st.success(f"Last run finished at **{train_status['finished_at']}**")
    else:
        st.write("No retraining has run yet.")
