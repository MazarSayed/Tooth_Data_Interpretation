from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import numpy as np

# === Load VGG16 Model ===
def load_vgg16_model(model_path="dental_v1_vgg16.h5"):
    return load_model(model_path)

# === Predict using VGG16 Model ===
def predict_with_vgg16(model, image_file, target_size=(256, 256)):
    img = Image.open(image_file).convert("RGB").resize(target_size)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    predicted_class = prediction.argmax()
    confidence = prediction[predicted_class]

    return predicted_class, confidence

# === (Optional) Class Labels Map ===
def get_class_label(index):
    label_map = {
        0: "Healthy",
        1: "Cavity",
        2: "Impacted Tooth",
        3: "Root Canal",
        4: "Unknown"
    }
    return label_map.get(index, "Unrecognized")
