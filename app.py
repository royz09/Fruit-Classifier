import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os
import time # Import time for st.spinner simulation

# Set page configuration
st.set_page_config(
    page_title="Fruit Classifier",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
        border: 3px solid #FFD93D;
    }
    .fruit-name {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        text-align: center;
    }
    .confidence-text {
        font-size: 1.5rem;
        text-align: center;
    }
    .confidence-bar {
        background: rgba(255,255,255,0.3);
        border-radius: 10px;
        margin: 10px 0;
        overflow: hidden;
    }
    .confidence-fill {
        background: linear-gradient(90deg, #FFD93D, #FF6B6B);
        height: 25px;
        border-radius: 10px;
        text-align: center;
        color: black;
        font-weight: bold;
        line-height: 25px;
    }
    .stats-card {
        background: rgba(255,255,255,0.1);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .fruit-card {
        background: rgba(255,255,255,0.05);
        padding: 15px;
        border-radius: 10px;
        margin: 8px 0;
        border-left: 4px solid;
    }
    .apple-card { border-left-color: #FF6B6B !important; }
    .banana-card { border-left-color: #FFD93D !important; }
    .orange-card { border-left-color: #FFA500 !important; }
    .grapes-card { border-left-color: #9370DB !important; }
    .strawberry-card { border-left-color: #FF4500 !important; }
</style>
""", unsafe_allow_html=True)

# Load the fruit information from JSON
@st.cache_data
def load_fruit_info(json_path='fruit_info.json'):
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"❌ Error: '{json_path}' not found. Please ensure it's in the same directory as app.py.")
        return {}

fruit_info_data = load_fruit_info()

# Load the TFLite model
@st.cache_resource
def load_tflite_model(model_path='fruit_classifier_quantized.tflite'):
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        # Get input and output details within the cached function
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        st.success("✅ Fruit Classifier (TFLite) Loaded!")
        return interpreter, input_details, output_details
    except FileNotFoundError:
        st.error(f"❌ Error: '{model_path}' not found. Please ensure the quantized model is in the same directory.")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading TFLite model: {e}")
        return None, None, None


# Function to preprocess the image
def preprocess_image(image, input_shape, input_dtype):
    img = image.resize((input_shape[1], input_shape[2]))
    img_array = np.array(img) # Get raw pixel values

    # Ensure 3 channels
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:  # RGBA
        img_array = img_array[:, :, :3]

    # Normalize and convert to appropriate dtype
    if input_dtype == tf.float32 or input_dtype == np.float32:
        img_array = img_array.astype(np.float32) / 255.0  # Normalize to [0, 1]
    elif input_dtype == tf.uint8 or input_dtype == np.uint8: # If model expects uint8 input (e.g., from full integer quantization)
        img_array = img_array.astype(np.uint8) # Keep as [0, 255]
    else:
        st.warning(f"Unexpected input dtype: {input_dtype}. Defaulting to float32 normalization.")
        img_array = img_array.astype(np.float32) / 255.0

    return np.expand_dims(img_array, axis=0) # Add batch dimension

# Function to make prediction (returns raw predictions array)
def predict_fruit(interpreter, input_details, output_details, image):
    # Debugging information
    st.write(f"DEBUG: Model expected input shape: {input_details[0]['shape']}")
    st.write(f"DEBUG: Model expected input dtype: {input_details[0]['dtype']}")

    processed_image = preprocess_image(image, input_details[0]['shape'], input_details[0]['dtype'])

    st.write(f"DEBUG: Processed image shape: {processed_image.shape}")
    st.write(f"DEBUG: Processed image dtype: {processed_image.dtype}")

    interpreter.set_tensor(input_details[0]['index'], processed_image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    return predictions

def get_fruit_emoji(fruit_name):
    """Get emoji for each fruit"""
    emoji_map = {
        'apple': '🍎',
        'banana': '🍌',
        'orange': '🍊',
        'grapes': '🍇',
        'strawberry': '🍓'
    }
    return emoji_map.get(fruit_name, '❓')

def main():
    st.markdown('<h1 class="main-header">🍎 Fruit Classifier</h1>', unsafe_allow_html=True)

    st.markdown("""
    ### Professional Fruit Identification
    Upload an image and our AI will identify the fruit with high accuracy!
    """)

    # Define the fruit classes (must match training order)
    fruit_classes = list(fruit_info_data.keys())

    # Load model and fruit info
    interpreter, input_details, output_details = load_tflite_model()

    if interpreter is None or not fruit_info_data:
        st.stop() # Stop if model or fruit info failed to load

    # Stats sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        st.markdown("""
        <div class="stats-card">
            <strong>Task:</strong> Multi-class Classification<br>
            <strong>Classes:</strong> 5 Fruits<br>
            <strong>Input Size:</strong> 128×128 pixels (as trained)<br>
            <strong>Architecture:</strong> CNN<br>
            <strong>Model File:</strong> < 25MB (TFLite quantized)<br>
            <strong>Optimization:</strong> Dynamic Range Quantization
        </div>
        """, unsafe_allow_html=True)

        st.header("🎯 Supported Fruits")
        for fruit in fruit_classes:
            emoji = get_fruit_emoji(fruit)
            info = fruit_info_data.get(fruit, {})
            display_name = info.get('name', fruit.title())
            st.write(f"{emoji} {display_name}")

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📸 Upload Fruit Image")
        uploaded_file = st.file_uploader(
            "Choose an image of a fruit",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of apple, banana, orange, grapes, or strawberry"
        )

        image = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)

    with col2:
        st.subheader("🍎 Fruit Guide")

        for fruit in fruit_classes:
            info = fruit_info_data.get(fruit, {})
            emoji = get_fruit_emoji(fruit)
            display_name = info.get('name', fruit.title())
            color_class = f"{fruit}-card"

            st.markdown(f"""
            <div class="fruit-card {color_class}">
                <strong>{emoji} {display_name}</strong><br>
                <small>Season: {info.get('season', 'N/A')}</small>
            </div>
            """, unsafe_allow_html=True)

    # Prediction
    if image is not None:
        st.markdown("---")
        st.subheader("🎯 Classification Results")

        predictions = predict_fruit(interpreter, input_details, output_details, image)

        # Get top predictions
        top_k = 3
        # Sort predictions and get indices of top-k
        top_indices = np.argsort(predictions)[::-1][:top_k]
        top_confidences = predictions[top_indices]
        top_fruits_names = [fruit_classes[i] for i in top_indices]

        # Display top prediction prominently
        st.markdown("### 🏆 Top Prediction")
        predicted_fruit = top_fruits_names[0]
        confidence = top_confidences[0]
        emoji = get_fruit_emoji(predicted_fruit)
        fruit_display_name = fruit_info_data.get(predicted_fruit, {}).get('name', predicted_fruit.title())

        st.markdown(f"""
        <div class="prediction-card">
            <div class="fruit-name">{emoji} {fruit_display_name}</div>
            <div class="confidence-text">Confidence: {confidence:.2%}</div>
        </div>
        """, unsafe_allow_html=True)

        # Display top 3 predictions with bars
        if top_k > 1:
            st.markdown("### 📈 Top 3 Confidence Scores")

            # Normalize confidences for display if needed (e.g., if sum != 1 due to TFLite quirks)
            # For softmax output, they should sum to 1.
            total_confidence = np.sum(top_confidences)
            if total_confidence > 0: # Avoid division by zero
                 normalized_top_confidences = top_confidences / total_confidence
            else:
                 normalized_top_confidences = top_confidences # Keep as is if all zero

            for i in range(len(top_fruits_names)):
                fruit_name = top_fruits_names[i]
                fruit_conf = top_confidences[i]
                fruit_emoji = get_fruit_emoji(fruit_name)
                fruit_info = fruit_info_data.get(fruit_name, {})
                display_name = fruit_info.get('name', fruit_name.title())

                col_name, col_bar = st.columns([3, 7]) # Adjusted column ratio for better layout
                with col_name:
                    st.write(f"{fruit_emoji} **{display_name}**")
                with col_bar:
                    # Use the actual confidence for the progress bar, not necessarily normalized to 1 across top-k
                    st.progress(float(fruit_conf))
                    st.caption(f"{fruit_conf:.2%}")

        # Display detailed information about the top predicted fruit
        with st.expander(f"📖 Learn more about {fruit_display_name}"):
            if predicted_fruit in fruit_info_data:
                info = fruit_info_data[predicted_fruit]
                st.write(f"**Description:** {info.get('description', 'N/A')}")
                st.write(f"**Characteristics:** {', '.join(info.get('characteristics', []))}")
                st.write(f"**Nutrition:** {info.get('nutrition', 'N/A')}")
                st.write(f"**Season:** {info.get('season', 'N/A')}")
                st.write(f"**Colors:** {', '.join(info.get('colors', []))}")
                st.write(f"**Fun Fact:** {info.get('fun_fact', 'N/A')}")
            else:
                st.info("No detailed information available for this fruit.")

if __name__ == "__main__":
    main()
