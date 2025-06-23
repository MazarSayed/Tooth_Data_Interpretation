import os
import re
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
# === Load API Key ===
def load_api_key(provider="openai"):
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
    return os.getenv("OPENAI_API_KEY") if provider == "openai" else os.getenv("GEMINI_API_KEY")

# === Save Uploaded Image ===
def save_uploaded_image(uploaded_file):
    img = Image.open(uploaded_file)
    img_path = "uploaded_image.png"
    img.save(img_path)
    return img, img_path

# === Run YOLO Detection ===
def run_yolo_detection(img_path, model_path="yolov8n.pt"):
    model = YOLO(model_path)
    results = model(img_path)
    results[0].save("annotated.png")
    return results[0], "annotated.png", model

# === Parse YOLO Results ===
def parse_yolo_output(results, model):
    detections = []
    for box in results.boxes:
        label = model.model.names[int(box.cls)]
        conf = float(box.conf)
        coords = box.xyxy
        if hasattr(coords, "tolist"):
            coords = coords.tolist()
            xyxy = [round(float(x), 2) for x in (coords[0] if isinstance(coords[0], list) else coords)]
        else:
            xyxy = [0, 0, 0, 0]
        detections.append(f"Detected {label} with confidence {conf:.2f} at {xyxy}")
    return "\n".join(detections) if detections else "No objects detected."

# === Get LLM Output ===
def get_llm_output(api_key, prompt_path, model_info):
    provider = model_info["provider"]
    model_name = model_info["model"]

    with open(prompt_path, "r", encoding="utf-8") as file:
        prompt = file.read()

    if provider == "openai":
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content

    elif provider == "gemini":
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Please install `google-generativeai` to use Gemini models.")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text

    else:
        raise ValueError(f"Unsupported provider: {provider}")


# === Extract Metadata from LLM Output ===
def extract_metadata(llm_output):
    scan_type, scan_tool, exam_date = "", "", ""
    for line in llm_output.splitlines():
        if "**Scan Type**" in line:
            scan_type = line.split("**Scan Type**:")[1].strip()
        elif "**Analytical Tool**" in line:
            scan_tool = line.split("**Analytical Tool**:")[1].strip()
        elif "**Date of Examination**" in line:
            exam_date = line.split("**Date of Examination**:")[1].strip()
    return scan_type, scan_tool, exam_date

# === Extract Table from LLM Output ===
def extract_table(llm_output):
    table_lines = []
    capture = False
    for line in llm_output.splitlines():
        if line.strip().startswith("|") and "Tooth" in line:
            capture = True
        if capture and line.strip().startswith("|"):
            table_lines.append(line.strip())
        elif capture and not line.strip().startswith("|"):
            break

    table_lines = [line for line in table_lines if not re.match(r"^\|[-| ]+\|$", line)]
    if not table_lines:
        return None

    headers = [h.strip() for h in table_lines[0].strip("|").split("|")]
    data_rows = table_lines[1:]
    data = []
    for row in data_rows:
        cols = [c.strip() for c in row.strip("|").split("|")]
        if len(cols) == len(headers):
            data.append(cols)

    return pd.DataFrame(data, columns=headers)
