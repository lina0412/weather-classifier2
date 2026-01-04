# streamlit_app.py - DEMO VERSION (No model required)
import streamlit as st
import numpy as np
from PIL import Image
import json
import random

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
    .demo-badge {
        background-color: #FF6B6B;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title with demo badge
st.markdown('<h1 class="main-title">⛈️ Weather Image Classifier <span class="demo-badge">DEMO MODE</span></h1>', unsafe_allow_html=True)
st.markdown("### Classify: Hail • Lightning • Rain • Sandstorm • Snow")

# Generate mock predictions (no model needed)
def generate_mock_predictions():
    """Generate realistic-looking random predictions"""
    class_names = ['hail', 'lightning', 'rain', 'sandstorm', 'snow']
    
    # Generate random probabilities that sum to 1
    probs = np.random.dirichlet(np.ones(5) * 2)
    
    # Sort to make one class more prominent (like a real model would)
    probs = np.sort(probs)[::-1]
    
    # Make the highest probability more realistic (40-95%)
    probs[0] = random.uniform(0.4, 0.95)
    
    # Renormalize
    probs = probs / probs.sum()
    
    return dict(zip(class_names, probs))

# Load mock class names
def load_class_names():
    return ['hail', 'lightning', 'rain', 'sandstorm', 'snow']

# Sidebar
with st.sidebar:
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a weather image",
        type=['jpg', 'jpeg', 'png'],
        help="Supported: JPG, JPEG, PNG"
    )
    
    st.header("📊 System Info")
    st.warning("⚠️ Running in **Demo Mode**")
    st.info("To use real AI predictions:")
    st.markdown("1. Upload model files to GitHub")
    st.markdown("2. Replace mock predictions")
    
    st.header("🎮 Demo Controls")
    
    # Add some interactivity
    confidence_level = st.slider(
        "Simulated Accuracy",
        min_value=0.5,
        max_value=0.95,
        value=0.75,
        help="Adjust how confident the mock predictions appear"
    )
    
    if st.button("🎲 Randomize Predictions"):
        st.rerun()

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
            st.write(f"**Mode:** {image.mode}")
    else:
        st.info("👈 Upload an image to get started")
        st.markdown("### Try uploading:")
        st.markdown("- ⛈️ Storm clouds")
        st.markdown("- ❄️ Snowy landscape")
        st.markdown("- 🌪️ Dust/sandstorm")
        st.markdown("- ⚡ Lightning photo")
        st.markdown("- 🌧️ Rainy scene")

with col2:
    st.header("🎯 Predictions")
    
    if uploaded_file:
        # Show loading animation
        with st.spinner('🔬 Simulating AI analysis...'):
            # Small delay to simulate processing
            import time
            time.sleep(1.5)
            
            # Generate mock predictions
            predictions = generate_mock_predictions()
            
            # Get highest prediction
            predicted_class = max(predictions, key=predictions.get)
            confidence = predictions[predicted_class]
            
            # Adjust based on slider
            confidence = min(confidence, confidence_level)
        
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
        st.metric("Simulated Confidence", f"{confidence:.2%}")
        st.caption("Note: Using mock predictions in demo mode")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # All predictions
        st.subheader("📊 All Simulated Probabilities")
        
        # Sort by confidence
        sorted_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        for class_name, prob in sorted_items:
            emoji = emoji_map.get(class_name, '⛈️')
            
            col_a, col_b, col_c = st.columns([1, 3, 1])
            with col_a:
                st.write(f"**{emoji} {class_name.title()}**")
            with col_b:
                st.progress(float(prob))
            with col_c:
                st.write(f"**{prob:.2%}**")
        
        # Additional stats
        with st.expander("📈 Prediction Statistics"):
            st.metric("Highest Confidence", f"{max(predictions.values()):.2%}")
            st.metric("Lowest Confidence", f"{min(predictions.values()):.2%}")
            st.metric("Confidence Range", f"{max(predictions.values()) - min(predictions.values()):.2%}")
        
        # Download results
        results = {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'all_predictions': predictions,
            'note': 'Generated in demo mode - not real AI predictions',
            'timestamp': str(np.datetime64('now'))
        }
        
        st.download_button(
            label="📥 Download Results (JSON)",
            data=json.dumps(results, indent=2),
            file_name="demo_prediction_results.json",
            mime="application/json"
        )
        
    else:
        # Welcome message
        st.markdown("""
        <div style="background: #f8f9fa; padding: 20px; border-radius: 10px;">
            <h3>🚀 How to Use (Demo Mode):</h3>
            <ol>
                <li><strong>Upload</strong> any weather-related image</li>
                <li><strong>Watch</strong> simulated AI analysis</li>
                <li><strong>View</strong> realistic mock predictions</li>
                <li><strong>Adjust</strong> settings in sidebar</li>
                <li><strong>Download</strong> sample results</li>
            </ol>
            
            <h3>🎯 Features:</h3>
            <ul>
                <li>No model files required</li>
                <li>Fully functional UI</li>
                <li>Realistic-looking predictions</li>
                <li>Adjustable confidence levels</li>
                <li>Export results as JSON</li>
            </ul>
            
            <div style="background: #FFF3CD; padding: 10px; border-radius: 5px; margin-top: 15px;">
                <strong>⚠️ Note:</strong> This is a demo. To use actual AI predictions, upload model files to your GitHub repository.
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("**Weather Classification Demo** | UI works without model files")
st.caption("To enable real AI: Add weather_classifier.keras to your project")

# Add a refresh button at the bottom
if st.button("🔄 Refresh Demo"):
    st.rerun()
