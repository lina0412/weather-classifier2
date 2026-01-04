# streamlit_app.py - FIXED VERSION
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

st.set_page_config(
    page_title="Weather Classification System",
    page_icon="⛈️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-card {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-title">⛈️ Weather Image Classifier</h1>', unsafe_allow_html=True)
st.markdown("### Classify: Hail • Lightning • Rain • Sandstorm • Snow")

# Load model with multiple fallbacks
@st.cache_resource
def load_model():
    """Try multiple model files"""
    model_files = [
        'weather_classifier_fixed.keras',  # Your fixed model
        'weather_classifier.keras',
        'weather_classifier_deployment.keras',
        'weather_classifier.h5'
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                model = tf.keras.models.load_model(model_file)
                st.sidebar.success(f"✓ Loaded: {model_file}")
                return model
            except Exception as e:
                st.sidebar.warning(f"Could not load {model_file}: {e}")
                continue
    
    # Fallback: create simple model
    st.sidebar.error("Using fallback model")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

# Load class names
@st.cache_data
def load_class_names():
    try:
        with open('class_names.json', 'r') as f:
            return json.load(f)
    except:
        return ['hail', 'lightning', 'rain', 'sandstorm', 'snow']

# Sidebar
with st.sidebar:
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a weather image",
        type=['jpg', 'jpeg', 'png'],
        help="Supported: JPG, JPEG, PNG"
    )
    
    st.header("📊 Model Info")
    st.write("**Classes:** 5 weather types")
    st.write("**Accuracy:** 70.56%")
    st.write("**Size:** 1.31 MB")
    st.write("**Compression:** 19.1x smaller")
    
    # Test model loading
    if st.button("Test Model Load"):
        try:
            model = load_model()
            class_names = load_class_names()
            st.success(f"✅ Model ready! Classes: {len(class_names)}")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("🖼️ Image Preview")
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with st.expander("Image Details"):
            st.write(f"**Size:** {image.size[0]} × {image.size[1]} pixels")
            st.write(f"**Format:** {image.format}")
    else:
        st.info("👈 Upload an image to get started")

with col2:
    st.header("🎯 Predictions")
    
    if uploaded_file:
        try:
            # Load model and classes
            model = load_model()
            class_names = load_class_names()
            
            with st.spinner('🔬 Analyzing image...'):
                # Preprocess
                img = Image.open(uploaded_file).resize((224, 224))
                img_array = np.array(img) / 255.0
                
                # Handle different image formats
                if len(img_array.shape) == 2:  # Grayscale
                    img_array = np.stack([img_array]*3, axis=-1)
                elif img_array.shape[2] == 4:  # RGBA
                    img_array = img_array[:, :, :3]
                
                img_array = np.expand_dims(img_array, axis=0)
                
                # Make prediction
                predictions = model.predict(img_array, verbose=0)[0]
                predicted_idx = np.argmax(predictions)
                predicted_class = class_names[predicted_idx]
                confidence = predictions[predicted_idx]
            
            # Display results
            emoji_map = {
                'hail': '🌨️',
                'lightning': '⚡', 
                'rain': '🌧️',
                'sandstorm': '🌪️',
                'snow': '❄️'
            }
            
            emoji = emoji_map.get(predicted_class, '⛈️')
            
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.success(f"### {emoji} {predicted_class.upper()} {emoji}")
            st.metric("Confidence", f"{confidence:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # All predictions
            st.subheader("📊 All Probabilities")
            
            # Sort by confidence
            sorted_indices = np.argsort(predictions)[::-1]
            
            for idx in sorted_indices:
                class_name = class_names[idx]
                prob = predictions[idx]
                emoji = emoji_map.get(class_name, '⛈️')
                
                col_a, col_b, col_c = st.columns([1, 3, 1])
                with col_a:
                    st.write(f"**{emoji} {class_name.title()}**")
                with col_b:
                    st.progress(float(prob))
                with col_c:
                    st.write(f"**{prob:.2%}**")
            
            # Download results
            results = {
                'predicted_class': predicted_class,
                'confidence': float(confidence),
                'all_predictions': dict(zip(class_names, predictions.tolist()))
            }
            
            st.download_button(
                label="📥 Download Results (JSON)",
                data=json.dumps(results, indent=2),
                file_name="prediction_results.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("Make sure all model files are in the same folder")
    else:
        # Welcome message
        st.markdown("""
        <div style="background: #f8f9fa; padding: 20px; border-radius: 10px;">
            <h3>🚀 How to Use:</h3>
            <ol>
                <li><strong>Upload</strong> a weather image</li>
                <li><strong>Wait</strong> for AI analysis (~2 seconds)</li>
                <li><strong>View</strong> predictions with confidence scores</li>
                <li><strong>Download</strong> results as JSON</li>
            </ol>
            
            <h3>🎯 Best Results With:</h3>
            <ul>
                <li>Clear weather images</li>
                <li>Good lighting</li>
                <li>Centered subject</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("**Weather Classification System** | Transfer Learning & Knowledge Distillation")
