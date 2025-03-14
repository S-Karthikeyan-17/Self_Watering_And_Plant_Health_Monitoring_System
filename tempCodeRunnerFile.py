import os
import numpy as np
from flask import Flask, request, jsonify, make_response
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model
model_path = 'plant_disease_model.h5'
model = tf.keras.models.load_model(model_path)

# Define the class labels
class_labels = ['healthy', 'multiple_diseases', 'rust', 'scab']

# Function to preprocess the image
def preprocess_image(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

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
    file.save(file_path)

    # Preprocess the image
    preprocessed_img = preprocess_image(file_path)

    # Make a prediction
    prediction = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class_index]

    # Get the predicted class label
    predicted_label = class_labels[predicted_class_index]

    # Print the prediction in the terminal
    print(f"Predicted label: {predicted_label}, Confidence: {confidence:.2f}")

    # Create the response with additional headers for security and performance
    response = make_response(jsonify({
        'category': predicted_label,
        'confidence': float(confidence)
    }))
    # Add security and caching headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Cache-Control'] = 'no-store, must-revalidate'
    response.headers['Content-Security-Policy'] = "default-src 'self'; frame-ancestors 'none';"
    response.headers['X-Frame-Options'] = 'DENY'  # Prevent clickjacking
    response.headers['Expires'] = '0'  # Ensure caching is disabled

    return response

if __name__ == '__main__':
    app.run(debug=True)
