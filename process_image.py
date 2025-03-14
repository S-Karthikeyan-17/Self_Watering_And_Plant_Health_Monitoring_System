from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS) for frontend communication

# Load the trained model
MODEL_PATH = "plant_disease_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define expected input image size (adjust according to your model)
IMAGE_SIZE = (224, 224)  # Change if your model expects a different size

@app.route('/predict', methods=['POST'])
def predict():
    print("📷 Received request to /predict")

    # Ensure an image is received in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image file received"}), 400

    file = request.files['image']
    print(f"✅ Received image: {file.filename}")

    try:
        # Load the image and preprocess
        image = Image.open(io.BytesIO(file.read()))
        image = image.resize(IMAGE_SIZE)  # Resize to match model input
        image_array = np.array(image) / 255.0  # Normalize pixel values
        image_array = image_array.reshape(1, 224, 224, 3)  # Ensure correct shape

        # Ensure image shape matches model input
        expected_shape = model.input_shape
        if image_array.shape[1:] != expected_shape[1:]:
            return jsonify({"error": f"Invalid image shape: {image_array.shape}, expected: {expected_shape}"}), 400

        # Make a prediction
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction)  # Get class with highest probability
        confidence = float(np.max(prediction))  # Get confidence score

        print(f"🔍 Prediction: Class {predicted_class} (Confidence: {confidence:.2f})")

        return jsonify({
            "prediction": int(predicted_class),
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
