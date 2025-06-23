import streamlit as st
from utils import (
    load_vgg16_model, predict_with_vgg16, get_class_label
)

st.set_page_config(page_title="VGG16 Dental Classifier", page_icon="🦷", layout="wide")
st.title("🧠 Dental X-ray Classification using VGG16")

# === File Upload ===
uploaded_file = st.file_uploader("Upload a dental X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Dental X-ray", width=400)

    # Load VGG16 model
    model = load_vgg16_model("dental_v1_vgg16.h5")

    # Run prediction
    label_index, confidence = predict_with_vgg16(model, uploaded_file)
    label = get_class_label(label_index)

    # Display results
    st.markdown("### 🧾 Prediction Result")
    st.markdown(f"**Predicted Class:** `{label}`")
    st.markdown(f"**Confidence:** `{confidence:.2f}`")

    st.warning("⚠️ This is an AI-generated prediction. Final interpretation should be done by a licensed dental professional.")
