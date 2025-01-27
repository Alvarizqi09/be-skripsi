from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})  # Configuring CORS for React front-end

# Load the trained model
model_path = "./Plant_Leaf_Model5.keras"  # Update with the correct path
model = tf.keras.models.load_model(model_path)

# Define class names (this should match the order in your model)
class_names = ["Corn Common Rust", "Corn Gray Leaf Spot", "Corn Healthy","Corn Northern Leaf Blight"]

def prepare_image(img_path, img_size=(299, 299)):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    return img_array

def predict_image(img_path, model):
    img_array = prepare_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence

@app.route("/predict", methods=["POST"])
def predict():
    try:
        img_file = request.files['image']
        img_path = f"/tmp/{img_file.filename}"  # Save image to a temporary location
        img_file.save(img_path)

        predicted_class, confidence = predict_image(img_path, model)

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": float(confidence)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
