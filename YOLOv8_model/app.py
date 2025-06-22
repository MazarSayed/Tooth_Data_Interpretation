import streamlit as st
import os
from utils import (
    load_api_key, save_uploaded_image, run_yolo_detection, parse_yolo_output,
    get_llm_output, extract_metadata, extract_table
)

st.set_page_config(page_title="Dental X-ray Insight Report", page_icon="🦷", layout="wide")
st.title("🦷 Dental Radiology AI Reporter")

openai_api_key = load_api_key()
uploaded_file = st.file_uploader("Upload a dental X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img, img_path = save_uploaded_image(uploaded_file)
    results, annotated_path, model = run_yolo_detection(img_path)
    yolo_text = parse_yolo_output(results, model)

    st.markdown("---")
    st.markdown("### 📸 Scan Overview")
    st.image(annotated_path, caption="YOLO Output", width=400)

    if openai_api_key:
        with st.spinner("Generating report with LLM..."):
            prompt_path = os.path.join("prompt", "dental_report_prompt.txt")
            llm_output = get_llm_output(openai_api_key, prompt_path)

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
        st.warning("🔐 OpenAI API key not found in `.env` file.")

