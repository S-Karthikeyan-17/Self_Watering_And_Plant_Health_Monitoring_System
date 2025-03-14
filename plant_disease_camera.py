import cv2
import numpy as np
import tensorflow as tf
import firebase_admin
from firebase_admin import credentials, storage, db
import time

# Load Firebase credentials
cred = credentials.Certificate("serviceAccountKey.json")  # Replace with your Firebase JSON key
firebase_admin.initialize_app(cred, {
    'storageBucket': "plant-monitoring-system-58336.firebasestorage.app",
    'databaseURL': 'https://plant-monitoring-system-58336-default-rtdb.asia-southeast1.firebasedatabase.app'
})

bucket = storage.bucket()

# Load trained ML model
model = tf.keras.models.load_model("plant_disease_model.h5")

# Define class labels
CLASS_LABELS = ["Healthy", "Rust", "Scab", "Multiple Diseases"]

# Open the laptop camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

print("Press 's' to capture an image and classify, or 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Display live feed
    cv2.imshow("Camera Feed", frame)

    # Wait for user input
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):  # Capture on 's' key
        # Preprocess the image for model
        resized_img = cv2.resize(frame, (128, 128))
        normalized_img = resized_img / 255.0
        input_array = np.expand_dims(normalized_img, axis=0)

        # Predict using the ML model
        predictions = model.predict(input_array)
        predicted_class = np.argmax(predictions[0])  # Get class index
        classification_result = CLASS_LABELS[predicted_class]  # Get class label

        print(f"Classification: {classification_result}")

        # Save classified image
        image_filename = f"classified_leaf_{int(time.time())}.jpg"
        cv2.imwrite(image_filename, frame)

        # Upload to Firebase Storage
        blob = bucket.blob("classified_image.jpg")  # Overwrites old image
        blob.upload_from_filename(image_filename)
        blob.make_public()  # Make image publicly accessible

        # Update classification result in Firebase
        db.reference("classification").update({"result": classification_result})

        print("Image and result uploaded to Firebase.")

    elif key == ord('q'):  # Quit on 'q' key
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
