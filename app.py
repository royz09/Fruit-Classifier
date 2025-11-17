import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# -------------------------------
# Load model
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fruit_classifier.h5")  # Make sure this is in the same folder

model = load_model()

# -------------------------------
# Class info
# -------------------------------
class_names = ["Apple", "Orange", "Banana"]

# -------------------------------
# Image preprocessing
# -------------------------------
def preprocess(img):
    # Resize to 128x128
    img = img.resize((128, 128))
    # Convert to RGB (handles grayscale or RGBA images)
    img = img.convert("RGB")
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🍎 Fruit Classifier")

uploaded_file = st.file_uploader("Upload an image of Apple, Orange, or Banana", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        x = preprocess(img)

        # Predict
        pred = model.predict(x)
        if pred.shape[-1] != len(class_names):
            st.error(f"Model output shape {pred.shape[-1]} does not match number of classes {len(class_names)}")
        else:
            pred = pred[0]
            predicted_class = class_names[int(np.argmax(pred))]

            st.subheader(f"🎯 Predicted Class: {predicted_class}")

            # Show confidence for each class
            st.markdown("**Confidence Scores:**")
            for i, name in enumerate(class_names):
                st.write(f"- {name}: {float(pred[i]):.2%}")
                st.progress(float(pred[i]))

    except Exception as e:
        st.error(f"An error occurred: {e}")
