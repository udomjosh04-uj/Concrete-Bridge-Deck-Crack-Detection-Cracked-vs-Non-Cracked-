# Streamlit file
import os
import requests
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Concrete Crack Screening", layout="centered")
st.title("Explainable Concrete Crack Screening Tool")
st.warning(
    "Research screening prototype. This tool does not determine structural safety, "
    "crack severity, or replace inspection by a qualified structural engineer."
)
MODEL_PATH = "best_model.keras"
MODEL_URL = "https://huggingface.co/Abasiofon001/concrete-crack-classifier/resolve/main/best_model.keras"
IMG_SIZE = 224

@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            with st.spinner("Downloading model..."):
                r = requests.get(MODEL_URL, timeout=60)
                r.raise_for_status()
                with open(MODEL_PATH, "wb") as f:
                    f.write(r.content)
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

model = load_model()
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype(np.float32) / 255.0
    pred = model.predict(arr[None, ...], verbose=0)[0, 0]
    label = "Cracked" if pred >= 0.5 else "Non-cracked"
    st.image(
        img,
        caption=f"Prediction: {label} (probability={pred:.3f})",
        use_container_width=True
    )
    st.info(
        "Confirm any flagged crack with a qualified inspector before taking any action."
    )
