import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Path to the model and image
model_path = 'plant_disease_model.h5'
uploads_folder = 'images'
uploaded_image_name = 'test.jpg'  # Change this to the actual file name

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Define the class labels (make sure these are in the same order as used during training)
class_labels = ['healthy', 'multiple_diseases', 'rust', 'scab']

# Function to preprocess the image
def preprocess_image(img_path, target_size=(128, 128)):
    # Load and resize the image
    img = image.load_img(img_path, target_size=target_size)
    
    # Convert to array and normalize
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    return img_array

# Test the model
def test_model():
    image_path = os.path.join(uploads_folder, uploaded_image_name)

    if not os.path.exists(image_path):
        print(f"Error: {image_path} does not exist.")
        return

    # Preprocess the image
    preprocessed_img = preprocess_image(image_path)

    # Make a prediction
    prediction = model.predict(preprocessed_img)

    # Interpret the prediction
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class_index]

    # Get the predicted class label
    predicted_label = class_labels[predicted_class_index]

    # Output the result
    print(f"Predicted label: {predicted_label}, Confidence: {confidence:.2f}")

# Run the test
if __name__ == "__main__":
    test_model()
