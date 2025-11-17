import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# Load the trained model
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fruit_classifier.h5")

model = load_model()

# -------------------------------
# Preprocess image
# -------------------------------
def preprocess(image):
    img = image.resize((128, 128))
    img = np.array(img)/255.0

    # Ensure 3 channels
    if img.ndim == 2:           # grayscale
        img = np.stack([img]*3, axis=-1)
    elif img.shape[-1] == 4:    # RGBA
        img = img[:, :, :3]

    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# -------------------------------
# Streamlit App
# -------------------------------
st.title("🍎 Fruit Classifier")
st.write("Upload an image of an Apple, Orange, or Banana.")

uploaded_file = st.file_uploader("Choose an image", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=300)

    x = preprocess(img)
    pred = model.predict(x)[0]  # prediction vector

    class_names = ["Apple","Orange","Banana"]

    # Show predicted class
    predicted_class = class_names[int(np.argmax(pred))]
    st.subheader(f"🎯 Predicted Class: {predicted_class}")

    # Show confidence for each class
    for i, name in enumerate(class_names):
        st.write(f"**{name} Confidence:** {pred[i]:.2%}")
        st.progress(float(pred[i]))  # cast to float for Streamlit
