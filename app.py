import os
import numpy as np
from flask import Flask, request, jsonify, make_response, send_from_directory
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model_path = 'plant_disease_model.h5'
try:
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded successfully from '{model_path}'")
except Exception as e:
    print(f"Error loading model: {e}")
    raise RuntimeError("Failed to load the model. Exiting...")

# Define the class labels
class_labels = ['healthy', 'multiple_diseases', 'rust', 'scab']

# Function to preprocess the image
def preprocess_image(img_path, target_size=(128, 128)):
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# Endpoint to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the file to the uploads folder
    uploads_folder = 'uploads'
    if not os.path.exists(uploads_folder):
        os.makedirs(uploads_folder)

    file_path = os.path.join(uploads_folder, 'uploaded_image.png')
    try:
        file.save(file_path)
    except Exception as e:
        return jsonify({'error': f'Error saving the file: {e}'}), 500

    # Preprocess the image
    preprocessed_img = preprocess_image(file_path)
    if preprocessed_img is None:
        return jsonify({'error': 'Error during image preprocessing'}), 500

    # Make a prediction
    try:
        prediction = model.predict(preprocessed_img)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][predicted_class_index]
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {e}'}), 500

    # Get the predicted class label
    predicted_label = class_labels[predicted_class_index]

    # Print the prediction in the terminal
    print(f"Predicted label: {predicted_label}, Confidence: {confidence:.2f}")

    # Create the response
    response = make_response(jsonify({
        'category': predicted_label,
        'confidence': float(confidence)
    }))

    # Add security headers
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Content-Security-Policy'] = "frame-ancestors 'none';"

    return response

# Serve the static reports.html file
@app.route('/')
def serve_reports():
    return send_from_directory('static', 'reports.html')

if __name__ == '__main__':
    # Run the Flask application
    app.run(debug=True, host='0.0.0.0', port=5000)
