import streamlit as st
from PIL import Image
import os
import openai
import pandas as pd
import re
from dotenv import load_dotenv

import torch
from transformers import ViTImageProcessor, ViTForImageClassification

# Load OpenAI API key
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
openai_api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai_api_key)

st.set_page_config(page_title="Dental X-ray Auto-Reporter", layout="wide")
st.title("Dental X-ray → Vision Transformer → LLM Report")

uploaded_file = st.file_uploader("Upload a dental X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    img = img.convert("RGB")
    st.image(img, caption="Input X-ray", width=400)
    img.save("uploaded_image.png")

    # === ViT Classification ===
    with st.spinner("Analyzing with Vision Transformer..."):
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
        predicted_label = model.config.id2label[predicted_class_idx]
        st.success(f"ViT Model Prediction: {predicted_label}")

    # === Prepare LLM Prompt ===
    st.markdown("---")
    st.markdown("### LLM Structured Report")
    prompt_path = os.path.join(os.path.dirname(__file__), 'prompt', 'dental_report_prompt.txt')
    print("DEBUG:before calling the prompt file")
    if openai_api_key:
        with open(prompt_path, "r", encoding="utf-8") as file:
            prompt = file.read()
        with st.spinner("Contacting LLM..."):
            response = client.chat.completions.create(
                model="gpt-4-0125-preview",  # or "gpt-3.5-turbo"
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            llm_output = response.choices[0].message.content
            print("DEBUG:hello")
        st.markdown("**LLM Structured Report:**")
        with st.expander("🔍 Raw LLM Output"):
            st.code(llm_output)

        # === Extract findings from LLM output ===
        matches = re.findall(
            r'(Deep Caries|Caries) detected with a confidence level of (\d+\.\d+) at coordinates \[(\d+\.\d+), (\d+\.\d+), (\d+\.\d+), (\d+\.\d+)\]',
            llm_output
        )
        print("Matches",matches)
        # Format into table
        table_data = []
        for i, (condition, confidence, x1, y1, x2, y2) in enumerate(matches):
            table_data.append({
                "Finding #": f"{i+1}",
                "Condition": condition,
                "Confidence": float(confidence),
                "Top-Left (x,y)": f"({x1}, {y1})",
                "Bottom-Right (x,y)": f"({x2}, {y2})",
            })

        # Show table
        if table_data:
            df = pd.DataFrame(table_data)
            st.markdown("### 🦷 Caries Detection Table")
            st.dataframe(df)
        else:
            st.warning("⚠️ No structured caries findings detected from the LLM output.")
    else:
        st.warning("OpenAI API key not found in environment. Please add OPENAI_API_KEY to your .env file.")

st.markdown("---")
st.caption("🚀 Pipeline: Input Image → Vision Transformer Model → Diagnosis Text → LLM → Structured Output")
