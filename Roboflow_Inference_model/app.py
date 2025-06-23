import streamlit as st
import os
from utils import (
    load_api_key, save_uploaded_image,
    run_roboflow_inference, parse_roboflow_output,
    get_llm_output, extract_metadata, extract_table,draw_roboflow_annotations
)

st.set_page_config(page_title="Dental X-ray Insight Report", page_icon="🦷", layout="wide")
st.title("🦷 Dental Radiology AI Reporter")

# === 1. Sidebar: LLM Selection ===
st.sidebar.markdown("### 🤖 Choose LLM Model")
llm_model = st.sidebar.selectbox(
    "Select a language model:",
    ["gpt-4o (OpenAI)", "gemini-2.5-pro (Google)", "gpt-3.5-turbo (OpenAI)"],
    index=0
)

# === 2. Map LLM name to backend details ===
model_mapping = {
    "gpt-4o (OpenAI)": {"provider": "openai", "model": "gpt-4o"},
    "gemini-2.5-pro (Google)": {"provider": "gemini", "model": "gemini-2.5-pro-preview-06-05"},
    "gpt-3.5-turbo (OpenAI)": {"provider": "openai", "model": "gpt-3.5-turbo"}
}
selected_model = model_mapping[llm_model]

# === 3. Load API Keys ===
api_key = load_api_key(provider=selected_model["provider"])
roboflow_api_key = load_api_key(provider="roboflow")

# === 4. Upload and Process Image ===
uploaded_file = st.file_uploader("Upload a dental X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img, img_path = save_uploaded_image(uploaded_file)

    # === Roboflow Detection ===
    with st.spinner("Analyzing with Roboflow model..."):
        rf_result = run_roboflow_inference(img_path, api_key=roboflow_api_key)
        detection_summary = parse_roboflow_output(rf_result)

        annotated_path = draw_roboflow_annotations(img_path, rf_result)
    st.markdown("---")
    st.markdown("### 📸 Scan Overview")
    st.image(annotated_path, caption="Roboflow Annotated Image", width=400)
    st.markdown(f"**Roboflow Detection Summary:**\n\n```\n{detection_summary}\n```")

    # === Generate LLM Report ===
    if api_key:
        with st.spinner(f"Generating report with {llm_model}..."):
            prompt_path = os.path.join("prompt", "dental_report_prompt.txt")
            llm_output = get_llm_output(api_key, prompt_path, selected_model)

        scan_type, scan_tool, exam_date = extract_metadata(llm_output)
        st.markdown("### 🦷 Patient Dental Report")
        st.markdown(f"**Scan Type:** {scan_type}  \n**Analytical Tool:** {scan_tool}  \n**Date of Examination:** {exam_date}")

        df_markdown = extract_table(llm_output)
        if df_markdown is not None:
            st.markdown("### 📋 Summary of Findings")
            st.dataframe(df_markdown)
            st.markdown("---")
            st.warning("⚠️ This AI-generated report is for informational purposes only. Final diagnosis and treatment decisions must be made by a qualified dental professional.")
        else:
            st.warning("⚠️ No markdown table found in LLM output.")
    else:
        st.warning("🔐 API key not found for selected model.")
