# -------------------------------
# fruit_classifier_app.py
# -------------------------------

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Fruit Classifier", layout="centered")

# -------------------------------
# Load model
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fruit_classifier.h5")  # your .h5 file

model = load_model()

# -------------------------------
# Fruit info JSON
# -------------------------------
fruit_info = {
    "Apple": {
        "description": "A sweet, crunchy fruit, usually red, green, or yellow.",
        "examples": ["Red Delicious", "Granny Smith", "Fuji"]
    },
    "Orange": {
        "description": "A juicy citrus fruit, typically orange in color.",
        "examples": ["Navel Orange", "Mandarin", "Blood Orange"]
    },
    "Banana": {
        "description": "A soft, elongated fruit with yellow peel.",
        "examples": ["Cavendish", "Red Banana", "Plantain"]
    }
}

class_names = ["Apple", "Orange", "Banana"]

# -------------------------------
# Image preprocessing
# -------------------------------
def preprocess(img):
    img = img.resize((128, 128))  # match your model input size
    img_array = np.array(img)
    if img_array.ndim == 2:  # grayscale → convert to 3 channels
        img_array = np.stack([img_array]*3, axis=-1)
    elif img_array.shape[-1] == 4:  # RGBA → RGB
        img_array = img_array[..., :3]
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------------------
# Main app
# -------------------------------
st.title("🍎 Fruit Classifier")
uploaded_file = st.file_uploader("Upload a fruit image", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    x = preprocess(img)
    
    # Predict
    pred = model.predict(x)
    
    if pred.shape[-1] != len(class_names):
        st.error(f"Model output shape {pred.shape[-1]} does not match number of classes {len(class_names)}")
    else:
        pred = pred[0]
        predicted_class = class_names[int(np.argmax(pred))]

        st.subheader(f"🎯 Predicted Class: {predicted_class}")

        # Confidence bars
        st.markdown("**Class Confidence:**")
        for i, name in enumerate(class_names):
            st.write(f"{name}: {pred[i]*100:.2f}%")
            st.progress(float(pred[i]))  # convert to native float

        # Display JSON info
        info = fruit_info[predicted_class]
        st.markdown(f"**Description:** {info['description']}")
        st.markdown("**Examples:**")
        for example in info['examples']:
            st.write(f"- {example}")
