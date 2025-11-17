import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import time
import json

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
    .grape-card { border-left-color: #9370DB !important; }
    .strawberry-card { border-left-color: #FF4500 !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the pre-trained fruit classifier (TFLite)"""
    try:
        interpreter = tf.lite.Interpreter(model_path='fruit_classifier_quantized.tflite')
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        st.success("✅ Fruit Classifier (TFLite) Loaded!")
        return interpreter, input_details, output_details
    except Exception as e:
        st.error(f"❌ Error loading TFLite model: {e}")
        return None, None, None

@st.cache_data
def load_fruit_info():
    """Load fruit information"""
    try:
        with open('fruit_info.json', 'r') as f:
            return json.load(f)
    except:
        return {}

def preprocess_image(image, input_details):
    """Preprocess the image for the model"""
    # Get input size and type from input_details
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    # Resize to model input size (e.g., 96x96)
    image = image.resize((input_shape[1], input_shape[2]))
    img_array = np.array(image)

    # Ensure 3 channels
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:  # RGBA
        img_array = img_array[:, :, :3]

    # Normalize and convert to appropriate dtype
    if input_dtype == np.float32:
        img_array = img_array.astype(np.float32) / 255.0
    elif input_dtype == np.uint8: # If fully quantized model expects uint8 input
        img_array = img_array.astype(np.uint8)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_fruit(_interpreter, input_details, output_details, image, fruits):
    """Predict fruit type from image using TFLite interpreter"""
    processed_image = preprocess_image(image, input_details)

    with st.spinner('🔍 Analyzing fruit characteristics...'):
        time.sleep(1.5)
        # Set input tensor
        _interpreter.set_tensor(input_details[0]['index'], processed_image)

        # Invoke the interpreter
        _interpreter.invoke()

        # Get output tensor
        predictions = _interpreter.get_tensor(output_details[0]['index'])

    # TFLite output will typically be float32 for dynamic range quantization
    return predictions[0]

def get_fruit_emoji(fruit_name):
    """Get emoji for each fruit"""
    emoji_map = {
        'apple': '🍎',
        'banana': '🍌',
        'orange': '🍊',
        'grapes': '🍇',
        'strawberry': '🍓'
    }
    return emoji_map.get(fruit_name, '🍎')

def main():
    st.markdown('<h1 class="main-header">🍎 Fruit Classifier</h1>', unsafe_allow_html=True)

    st.markdown("""
    ### Professional Fruit Identification
    Upload an image and our AI will identify the fruit with high accuracy!
    """)

    # Define fruits
    fruits = ['apple', 'banana', 'orange', 'grapes', 'strawberry']

    # Load model and fruit info
    fruit_info = load_fruit_info()
    interpreter, input_details, output_details = load_model()

    if interpreter is None:
        st.error("Model failed to load. Please check the model file.")
        return

    # Stats sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        st.markdown("""
        <div class="stats-card">
            <strong>Task:</strong> Multi-class Classification<br>
            <strong>Classes:</strong> 5 Fruits<br>
            <strong>Input Size:</strong> 96×96 pixels<br>
            <strong>Architecture:</strong> Advanced CNN<br>
            <strong>Model File:</strong> < 25MB (TFLite)<br>
            <strong>Accuracy:</strong> High on trained patterns
        </div>
        """, unsafe_allow_html=True)

        st.header("🎯 Supported Fruits")
        for fruit in fruits:
            emoji = get_fruit_emoji(fruit)
            info = fruit_info.get(fruit, {})
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
            st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.subheader("🍎 Fruit Guide")

        for fruit in fruits:
            info = fruit_info.get(fruit, {})
            emoji = get_fruit_emoji(fruit)
            display_name = info.get('name', fruit.title())
            color_class = f"{fruit}-card"

            st.markdown(f"""
            <div class="fruit-card {color_class}">
                <strong>{emoji} {display_name}</strong><br>
                <small>Color: {info.get('colors', 'Various')}</small>
            </div>
            """, unsafe_allow_html=True)

    # Prediction
    if image is not None:
        st.markdown("---")
        st.subheader("🎯 Classification Results")

        predictions = predict_fruit(interpreter, input_details, output_details, image, fruits)

        # Get top predictions
        top_k = 3
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        top_fruits = [fruits[i] for i in top_indices]
        top_confidences = [predictions[i] for i in top_indices]

        # Display top prediction
        st.markdown("### 🏆 Top Prediction")
        top_fruit = top_fruits[0]
        top_confidence = top_confidences[0]
        emoji = get_fruit_emoji(top_fruit)
        info = fruit_info.get(top_fruit, {})
        display_name = info.get('name', top_fruit.title())

        st.markdown(f"""
        <div class="prediction-card">
            <div class="fruit-name">{emoji} {display_name}</div>
            <div class="confidence-text">Confidence: {top_confidence:.2%}</div>
        </div>
        """, unsafe_allow_html=True)

        # Top 3 predictions
        st.markdown("### 📈 Top 3 Predictions")

        for i, (fruit, confidence) in enumerate(zip(top_fruits, top_confidences)):
            info = fruit_info.get(fruit, {})
            emoji = get_fruit_emoji(fruit)
            display_name = info.get('name', fruit.title())

            col1, col2 = st.columns([3, 2])
            with col1:
                st.write(f"**{i+1}. {emoji} {display_name}**")
            with col2:
                st.write(f"{confidence:.2%}")

            progress_html = f"""
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence*100}%">
                    {confidence:.2%}
                </div>
            </div>
            """
            st.markdown(progress_html, unsafe_allow_html=True)

        # Fruit information
        with st.expander("📖 Fruit Information"):
            if top_fruit in fruit_info:
                info = fruit_info[top_fruit]
                st.write(f"### About {info.get('name', top_fruit.title())}")
                st.write(f"**Description:** {info.get('description', 'N/A')}")

                st.write("**Characteristics:**")
                for feature in info.get('characteristics', []):
                    st.write(f"- {feature}")

                st.write(f"**Nutrition:** {info.get('nutrition', 'N/A')}")
                st.write(f"**Season:** {info.get('season', 'N/A')}")
                st.write(f"**Fun Fact:** {info.get('fun_fact', 'N/A')}")

if __name__ == "__main__":
    main()
