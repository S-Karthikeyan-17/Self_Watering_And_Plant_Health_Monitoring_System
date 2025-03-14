import tensorflow as tf
import numpy as np
import firebase_admin
from firebase_admin import credentials, db
from PIL import Image
import io
import base64

# Initialize Firebase
cred = credentials.Certificate("firebase_credentials.json")  # Your Firebase service account key
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://your-database-name.firebaseio.com/'  # Replace with your Firebase URL
})

# Load the trained ML model
model = tf.keras.models.load_model("plant_disease_model.h5")

# Class labels
class_names = ['healthy', 'multiple_diseases', 'rust', 'scab']

def preprocess_image(image_bytes):
    """ Preprocess the image for ML model input """
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((128, 128))
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict_disease(image_base64):
    """ Predict plant disease and upload to Firebase """
    image_bytes = base64.b64decode(image_base64)  # Decode base64 string
    image = preprocess_image(image_bytes)

    # Run model prediction
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    predicted_label = class_names[predicted_class]

    # Upload result to Firebase
    ref = db.reference('/predictions')
    ref.push({
        "prediction": predicted_label,
        "confidence": float(predictions[0][predicted_class])
    })

    return predicted_label

# Test Prediction (Optional)
if __name__ == "__main__":
    with open("test_leaf.jpg", "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        print("Prediction:", predict_disease(image_base64))
