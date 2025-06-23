import os
from PIL import Image
from dotenv import load_dotenv

# === Load API Key ===
def load_api_key(provider="gemini"):
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
    return os.getenv("GEMINI_API_KEY")

# === Save Uploaded Image ===
def save_uploaded_image(uploaded_file):
    img = Image.open(uploaded_file)
    img_path = "uploaded_image.png"
    img.save(img_path)
    return img, img_path

# === Generate Output with Gemini 2.5 Vision ===
def get_gemini_output(api_key, prompt, image):
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content([image, prompt])
    return response.text
