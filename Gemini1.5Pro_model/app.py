import streamlit as st
import os
from utils import load_api_key, save_uploaded_image, get_gemini_output
from datetime import date

st.set_page_config(page_title="Gemini Dental X-ray Reporter", page_icon="🦷", layout="wide")
st.title("🧠 Gemini 1.5 Pro Dental AI Reporter")

# === Load Gemini API Key ===
api_key = load_api_key(provider="gemini")

# === Upload Image ===
uploaded_file = st.file_uploader("Upload a dental X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img, img_path = save_uploaded_image(uploaded_file)
    st.image(img, caption="Uploaded Image", width=400)
    model_name = "gemini-1.5-pro"
    # === Load Prompt from File ===
    try:
        with open(os.path.join("prompt", "dental_report_prompt.txt"), "r", encoding="utf-8") as f:
            prompt = f.read()
            prompt = prompt.replace("{MODEL_NAME}", model_name)
    except FileNotFoundError:
        st.error("Prompt file not found in /prompt/dental_report_prompt.txt")
        prompt = ""

    # === Generate LLM Report ===
    if api_key and prompt:
        with st.spinner("Analyzing with Gemini 2.5..."):
            report = get_gemini_output(api_key, prompt, img)
        st.markdown("### 🦷 Gemini 1.5 Pro Report:")
        st.write(report)
        st.markdown("---")
        st.warning("⚠️ This is an AI-generated report. Please consult a dental professional for a clinical opinion.")
    elif not api_key:
        st.error("API key for Gemini not found.")
