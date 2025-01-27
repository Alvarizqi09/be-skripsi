import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)

# Configure CORS (adjust according to your needs)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})  # React frontend

# Enable logging
logging.basicConfig(level=logging.DEBUG)

# Load the trained model
model_path = "./Plant_Leaf_Model14.keras"  # Update this path if needed
model = tf.keras.models.load_model(model_path)

# Define class names (ensure this matches the model's output order)
class_names = ["Corn Common Rust", "Corn Gray Leaf Spot", "Corn Healthy", "Corn Northern Leaf Blight"]

def prepare_image(img_path, img_size=(299, 299)):
    try:
        img = image.load_img(img_path, target_size=img_size)
        img_array = image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
        return img_array
    except Exception as e:
        logging.error(f"Error in preparing image: {e}")
        raise

def predict_image(img_path, model):
    img_array = prepare_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the image file from the request
        img_file = request.files['image']
        
        # Determine the temporary directory path
        temp_dir = "/tmp" if os.path.exists("/tmp") else os.getcwd()  # Fallback to current directory
        img_path = os.path.join(temp_dir, img_file.filename)
        
        # Save the image to the temporary path
        img_file.save(img_path)

        # Make prediction
        predicted_class, confidence = predict_image(img_path, model)

        # Return the prediction as JSON response
        return jsonify({
            "predicted_class": predicted_class,
            "confidence": float(confidence)
        })

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
