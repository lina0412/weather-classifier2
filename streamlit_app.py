import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

st.set_page_config(page_title="Weather Classifier", layout="wide")

st.title("⛈️ Weather Image Classifier")
st.write("Classify: Hail • Lightning • Rain • Sandstorm • Snow")

# Sidebar
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose weather image", type=['jpg', 'png', 'jpeg'])
    
    st.header("Model Info")
    st.write("Distilled CNN Model")
    st.write("Accuracy: 70.56%")
    st.write("Size: 1.31 MB")
    st.write("19.1x smaller than teacher")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Uploaded Image")
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
    else:
        st.info("Please upload an image")

with col2:
    st.subheader("Predictions")
    if uploaded_file:
        # Load model
        try:
            model = tf.keras.models.load_model('weather_classifier_deployment.keras')
            with open('class_names.json', 'r') as f:
                class_names = json.load(f)
            
            # Preprocess
            img = Image.open(uploaded_file).resize((224, 224))
            img_array = np.array(img) / 255.0
            
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array]*3, axis=-1)
            elif img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]
            
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            with st.spinner('Analyzing...'):
                predictions = model.predict(img_array, verbose=0)[0]
            
            predicted_idx = np.argmax(predictions)
            predicted_class = class_names[predicted_idx]
            confidence = predictions[predicted_idx]
            
            # Display
            st.success(f"## {predicted_class.upper()}")
            st.metric("Confidence", f"{confidence:.2%}")
            
            # All predictions
            st.subheader("All Probabilities")
            for i, (name, prob) in enumerate(zip(class_names, predictions)):
                st.progress(float(prob), text=f"{name}: {prob:.2%}")
                
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Make sure model files exist in the same folder")
    else:
        st.info("Upload an image to see predictions")

st.markdown("---")
st.write("Weather Classification System | Transfer Learning Project")
