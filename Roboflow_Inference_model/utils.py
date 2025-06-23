# === utils.py ===

import os
import re
from PIL import ImageDraw, ImageFont
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from dotenv import load_dotenv
from openai import OpenAI
from inference_sdk import InferenceHTTPClient

# === Load API Key ===
def load_api_key(provider="openai"):
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
    if provider == "roboflow":
        return os.getenv("ROBOFLOW_API_KEY")
    return os.getenv("OPENAI_API_KEY") if provider == "openai" else os.getenv("GEMINI_API_KEY")

# === Save Uploaded Image ===
def save_uploaded_image(uploaded_file):
    img = Image.open(uploaded_file)
    img_path = "uploaded_image.png"
    img.save(img_path)
    return img, img_path

# === Run Roboflow Detection ===
def run_roboflow_inference(img_path, model_id="dental-j1vge/12", api_key=None):
    if not api_key:
        load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
        api_key = os.getenv("ROBOFLOW_API_KEY")

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key
    )
    result = client.infer(img_path, model_id=model_id)
    return result

# === Parse Roboflow Results ===
def parse_roboflow_output(result):
    detections = []
    predictions = result.get("predictions", [])
    for pred in predictions:
        label = pred.get("class", "Unknown")
        conf = float(pred.get("confidence", 0.0))
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        x1 = round(x - w / 2, 2)
        y1 = round(y - h / 2, 2)
        x2 = round(x + w / 2, 2)
        y2 = round(y + h / 2, 2)
        detections.append(f"Detected {label} with confidence {conf:.2f} at [{x1}, {y1}, {x2}, {y2}]")
    return "\n".join(detections) if detections else "No objects detected."

from PIL import ImageDraw, ImageFont, Image
def draw_roboflow_annotations(img_path, result, output_path="annotated_roboflow.png"):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except:
        font = ImageFont.load_default()

    for pred in result.get("predictions", []):
        label = pred.get("class", "Unknown")
        conf = float(pred.get("confidence", 0.0))
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        x1, y1 = x - w / 2, y - h / 2
        x2, y2 = x + w / 2, y + h / 2

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline="cyan", width=3)

        # Get text size using bbox
        label_text = f"{label} {conf:.2f}"
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Draw background rectangle for text
        draw.rectangle(
            [x1, y1 - text_height - 6, x1 + text_width + 6, y1],
            fill="cyan"
        )

        # Draw text
        draw.text((x1 + 3, y1 - text_height - 3), label_text, fill="black", font=font)

    img.save(output_path)
    return output_path

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
