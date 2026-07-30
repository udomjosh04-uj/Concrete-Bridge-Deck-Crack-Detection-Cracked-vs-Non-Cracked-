import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = "models/efficientnet_transfer_best.keras"
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
CLASS_NAMES = ["Non-cracked", "Cracked"]  # label flip applied during training: 0=Non-cracked, 1=Cracked
CONFIDENCE_FLAG_THRESHOLD = 0.65


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    array = np.array(image, dtype=np.float32)
    array = tf.keras.applications.efficientnet.preprocess_input(array)
    return np.expand_dims(array, axis=0)


def predict(model, image: Image.Image):
    batch = preprocess_image(image)
    prob_cracked = float(model.predict(batch, verbose=0)[0][0])
    prob_noncracked = 1.0 - prob_cracked

    if prob_cracked >= 0.5:
        predicted_class = CLASS_NAMES[1]
        confidence = prob_cracked
    else:
        predicted_class = CLASS_NAMES[0]
        confidence = prob_noncracked

    return predicted_class, confidence, {
        CLASS_NAMES[0]: prob_noncracked,
        CLASS_NAMES[1]: prob_cracked,
    }


def main():
    st.set_page_config(page_title="Concrete Deck Crack Classifier", layout="centered")
    st.title("Concrete Deck Crack Classifier")
    st.write("Classifies a bridge deck concrete image as Cracked or Non-cracked.")

    try:
        model = load_model()
    except Exception as e:
        st.error(f"Could not load model from '{MODEL_PATH}': {e}")
        st.stop()

    uploaded_file = st.file_uploader("Upload a concrete deck image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is None:
        st.stop()

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Running classification..."):
        predicted_class, confidence, class_probs = predict(model, image)

    st.subheader("Result")
    st.write(f"Predicted class: **{predicted_class}**")
    st.write(f"Confidence: **{confidence:.1%}**")

    if confidence < CONFIDENCE_FLAG_THRESHOLD:
        st.warning("Low confidence prediction.")

    st.subheader("Class probabilities")
    st.bar_chart(class_probs)

    with st.expander("Details"):
        st.write(f"Input size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
        st.write(f"Model: {MODEL_PATH}")
        st.json({k: round(v, 4) for k, v in class_probs.items()})


if __name__ == "__main__":
    main()
