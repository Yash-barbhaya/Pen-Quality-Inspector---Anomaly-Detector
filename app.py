import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Title of the app
st.title("🖊️ Pen Quality Inspector - Anomaly Detector")

# Load the trained Keras model
model = tf.keras.models.load_model("converted_keras/model.h5")

# Load class labels
with open("converted_keras/labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image of a pen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    # Show results
    st.subheader(f"🔍 Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")
