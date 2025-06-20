import streamlit as st
from ultralytics import YOLO
from PIL import Image
from dotenv import load_dotenv
import os
import openai

# Load OpenAI API key from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai_api_key)

st.set_page_config(page_title="Dental X-ray Auto-Reporter", layout="wide")
st.title("Dental X-ray → AI Model → LLM Report")

# Upload image
uploaded_file = st.file_uploader("Upload a dental X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    img_path = "uploaded_image.png"
    img.save(img_path)

    # Run YOLO
    model = YOLO("best.pt")
    results = model(img_path)
    results[0].save("annotated.png")
    st.image("annotated.png", caption="YOLO Output", width=400)

    # Get detection results as text
    detections = []
    for box in results[0].boxes:
        label = model.model.names[int(box.cls)]
        conf = float(box.conf)
        # Safely handle tensor shape for bounding box
        coords = box.xyxy
        if hasattr(coords, "tolist"):
            coords = coords.tolist()
            if isinstance(coords[0], list):  # Shape [1, 4]
                xyxy = [round(float(x), 2) for x in coords[0]]
            else:  # Shape [4]
                xyxy = [round(float(x), 2) for x in coords]
        else:
            xyxy = [0, 0, 0, 0]
        detections.append(f"Detected {label} with confidence {conf:.2f} at {xyxy}")

    yolo_text = "\n".join(detections) if detections else "No objects detected."

    
    # LLM Structured Report Section
    st.markdown("---")
    st.markdown("### LLM Structured Report")

    if openai_api_key and yolo_text != "No objects detectsed.":
        prompt = f"""
A panoramic dental X-ray was analyzed using an AI model.

Findings:
{yolo_text}

Please generate a structured dental radiology report based on the findings above.
"""
        openai.api_key = openai_api_key
        with st.spinner("Contacting LLM..."):
            response = client.chat.completions.create(
                model="gpt-4o",  # or "gpt-3.5-turbo"
                messages=[{"role": "user", "content": prompt}]
            )
            llm_output = response.choices[0].message.content
        st.markdown("**LLM Structured Report:**")
        st.success(llm_output)
    elif yolo_text == "No objects detected.":
        st.info("No objects detected to send to the LLM.")
    elif not openai_api_key:
        st.warning("OpenAI API key not found in environment. Please add OPENAI_API_KEY to your .env file.")

st.markdown("---")
st.caption("🚀 Pipeline: Input Image → YOLOv8 Model → Detection Text → LLM → Structured Output")
