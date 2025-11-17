import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fruit_classifier.h5")

model = load_model()

# -------------------------------
# Class Info
# -------------------------------
class_info = {
    "Apple": {
        "description": "A sweet, edible fruit produced by an apple tree.",
        "examples": ["Red apples", "Green apples", "Fuji apples"]
    },
    "Orange": {
        "description": "A citrus fruit known for its vibrant color and tangy taste.",
        "examples": ["Navel orange", "Mandarin", "Blood orange"]
    },
    "Banana": {
        "description": "A long curved fruit with soft sweet flesh and yellow skin.",
        "examples": ["Cavendish banana", "Plantain"]
    }
}

# -------------------------------
# Preprocess Image
# -------------------------------
def preprocess(image):
    img = image.resize((128,128))
    img = np.array(img)/255.0

    # Ensure 3 channels
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    elif img.shape[-1] == 4:
        img = img[:, :, :3]

    img = np.expand_dims(img, axis=0)
    return img

# -------------------------------
# App UI
# -------------------------------
st.set_page_config(page_title="🍎🍊🍌 Fruit Classifier", layout="wide")
st.title("🍎🍊🍌 Fruit Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

# Sidebar with class info
with st.sidebar:
    st.header("🍇 Fruit Info")
    for fruit, info in class_info.items():
        st.subheader(fruit)
        st.write(f"**Description:** {info['description']}")
        st.write("**Examples:**")
        for example in info["examples"]:
            st.write(f"- {example}")

# -------------------------------
# Prediction
# -------------------------------
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    x = preprocess(img)
    
    pred = model.predict(x)[0]
    class_names = ["Apple","Orange","Banana"]
    predicted_class = class_names[np.argmax(pred)]

    st.subheader(f"🎯 Predicted Class: {predicted_class}")

    # Confidence bars
    for i, name in enumerate(class_names):
        st.write(f"**{name} Confidence:** {pred[i]:.2%}")
        st.progress(pred[i])
