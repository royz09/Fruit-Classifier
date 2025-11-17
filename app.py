
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("fruit_classifier.h5")

st.title("🍎🍊🍌 Fruit Classifier")
uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

def preprocess(image):
    img = image.resize((128,128))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)
    return img

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    x = preprocess(img)
    pred = model.predict(x)[0]
    class_names = ["Apple","Orange","Banana"]
    for i, name in enumerate(class_names):
        st.write(f"{name}: {pred[i]:.2%}")
    st.write(f"**Predicted Class:** {class_names[np.argmax(pred)]}")
