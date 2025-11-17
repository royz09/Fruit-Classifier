import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os

# Load the fruit information from JSON
@st.cache_data
def load_fruit_info(json_path='fruit_info.json'):
    with open(json_path, 'r') as f:
        return json.load(f)

fruit_info_data = load_fruit_info()

# Load the TFLite model
@st.cache_resource
def load_tflite_model(model_path='fruit_classifier_quantized.tflite'):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# Define the fruit classes (must match training order)
fruit_classes = list(fruit_info_data.keys())

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((input_shape[1], input_shape[2]))
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    return np.expand_dims(img_array, axis=0) # Add batch dimension

# Function to make prediction
def predict_fruit(image):
    processed_image = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], processed_image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    # Get the class with the highest probability
    predicted_class_idx = np.argmax(predictions)
    predicted_fruit = fruit_classes[predicted_class_idx]
    confidence = np.max(predictions)
    return predicted_fruit, confidence

# Streamlit App Layout
st.title("Fruit Classifier")
st.write("Upload an image of a fruit (apple, banana, orange, grapes, strawberry) to classify it!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    predicted_fruit, confidence = predict_fruit(image)

    st.success(f"Prediction: {fruit_info_data[predicted_fruit]['name']} (Confidence: {confidence:.2f})")
    
    # Display fruit information
    if predicted_fruit in fruit_info_data:
        info = fruit_info_data[predicted_fruit]
        st.subheader(f"About {info['name']}")
        st.write(f"**Description:** {info['description']}")
        st.write(f"**Characteristics:** {', '.join(info['characteristics'])}")
        st.write(f"**Nutrition:** {info['nutrition']}")
        st.write(f"**Season:** {info['season']}")
        st.write(f"**Colors:** {', '.join(info['colors'])}")
        st.write(f"**Fun Fact:** {info['fun_fact']}")
    else:
        st.info("No detailed information available for this fruit.")
