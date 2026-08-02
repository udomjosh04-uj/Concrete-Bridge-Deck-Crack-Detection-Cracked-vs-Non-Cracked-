import numpy as np
import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image

# ---------------------------------------------------------------------------
# CONFIG — edit these to match your actual HF repo
# ---------------------------------------------------------------------------
HF_REPO_ID = "Abasiofon001/Concrete_Crack_Screening"   # <-- change this
HF_FILENAME = "best_model.keras"               # <-- change if you named it differently
IMG_SIZE = (224, 224)                          # must match training
LABEL_MAP = {0: "Non-cracked", 1: "Cracked"}   # must match training LABEL_MAP
DEFAULT_THRESHOLD = 0.5  # placeholder — replace with the threshold you picked
                          # from your validation PR curve, not left at 0.5 blind


# ---------------------------------------------------------------------------
# Model loading — cached so it only downloads/loads once per session, not
# on every rerun (Streamlit reruns the whole script on every interaction)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Downloading model from Hugging Face...")
def load_model():
    try:
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    except Exception as e:
        st.error(
            f"Could not download model from '{HF_REPO_ID}/{HF_FILENAME}'. "
            f"Check the repo id, filename, and that the repo is public "
            f"(or that you've set HF_TOKEN if it's private). Error: {e}"
        )
        st.stop()

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(
            f"Model file downloaded but failed to load. This is usually a "
            f"TensorFlow/Keras version mismatch between training and this "
            f"environment — check requirements.txt pins the same major TF "
            f"version you trained with. Error: {e}"
        )
        st.stop()

    return model


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """Must exactly mirror the training pipeline's preprocessing.
    Training used: decode -> resize -> cast float32 -> /255.0
    (EfficientNetV2 preprocess_input is applied INSIDE the model itself,
    so do NOT apply it again here or you'll double-preprocess.)"""
    img = pil_img.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)  # add batch dim


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Bridge Deck Crack Detector", page_icon="🌉")
st.title("🌉 Concrete Bridge Deck Crack Detector")
st.caption("EfficientNetV2-B0 transfer-learning model, Cracked vs Non-cracked.")

with st.spinner("Loading model..."):
    model = load_model()

threshold = st.slider(
    "Decision threshold (probability of 'Cracked' above which it's flagged)",
    min_value=0.0, max_value=1.0, value=DEFAULT_THRESHOLD, step=0.01,
    help=(
        "Lower this if you'd rather over-flag and manually review than miss "
        "a real crack. Don't leave this at the default without checking your "
        "own validation PR curve for the threshold that maximizes recall on "
        "the Cracked class."
    ),
)

uploaded_file = st.file_uploader(
    "Upload a bridge deck image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img, caption="Uploaded image", use_container_width=True)

    with st.spinner("Running inference..."):
        x = preprocess_image(pil_img)
        prob_cracked = float(model.predict(x, verbose=0).ravel()[0])

    pred_label = LABEL_MAP[1] if prob_cracked >= threshold else LABEL_MAP[0]

    with col2:
        if pred_label == "Cracked":
            st.error(f"**Prediction: {pred_label}**")
        else:
            st.success(f"**Prediction: {pred_label}**")

        st.metric("P(Cracked)", f"{prob_cracked:.3f}")
        st.progress(prob_cracked)
        st.caption(f"Threshold in use: {threshold:.2f}")

        if abs(prob_cracked - threshold) < 0.1:
            st.warning(
                "This prediction is close to the decision boundary — "
                "treat it as borderline, not confident, and consider manual review."
            )
else:
    st.info("Upload an image to run the model.")

st.divider()
st.caption(
    "This model was trained on the SDNET2018 Deck subset with a 2:1 "
    "(Non-cracked:Cracked) undersampled, group-leakage-checked split. "
    "It has not been validated on deck imagery outside that distribution "
    "(different concrete texture, lighting, camera angle, resolution) — "
    "treat out-of-distribution predictions with appropriate skepticism."
)
