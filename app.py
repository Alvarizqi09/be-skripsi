from waitress import serve
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import threading
import os
import gdown
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app, supports_credentials=True)  

MODEL_URL = "https://drive.google.com/uc?export=download&id=1-ZvSgcMMJBABNWk8TVjOiL_C3WZzbRcN"
MODEL_PATH = "/app/Plant_Leaf_ModelA6.keras"

# Cek apakah model sudah ada, jika belum, unduh
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)
    print("Download complete!")
    
# 📌 Load model setelah diunduh
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# 📌 Class names
class_names = ["Corn Common Rust", "Corn Gray Leaf Spot", "Corn Healthy", "Corn Northern Leaf Blight"]

def prepare_image(img_path, img_size=(299, 299)):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  
    return img_array

def predict_image(img_path, model):
    img_array = prepare_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Convert the predicted image to base64 string
    predicted_img = Image.open(img_path)
    buffered = io.BytesIO()
    predicted_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return predicted_class, confidence, img_str

@app.route("/predict", methods=["POST"])
def predict():
    try:
        img_file = request.files['image']
        img_path = f"/tmp/{img_file.filename}"  # Save image to a temporary location
        img_file.save(img_path)

        predicted_class, confidence, img_str = predict_image(img_path, model)

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "predicted_image": f"data:image/png;base64,{img_str}"  # Send base64 string for the predicted image
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

def run_flask():
    serve(app, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    threading.Thread(target=run_flask).start()
