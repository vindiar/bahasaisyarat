import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="SIBI Sign Language Detection System",
    page_icon="🤟",
    layout="wide"
)

# Load model (cached so it loads only once)
@st.cache_resource
def load_model():
    return YOLO("model/my_model.pt")

model = load_model()

# Header & Description
st.title("🤟 SIBI Sign Language Detection System")
st.write("A web-based system designed to detect SIBI sign language using YOLOv8, equipped with Text-to-Gesture translation.")

# Main Tabs menu
tab1, tab2, tab3 = st.tabs(["Text to Gesture", "Image Upload Detection", "Webcam Detection"])

# ==========================================
# TAB 1: TEXT TO GESTURE
# ==========================================
with tab1:
    st.header("Text to Gesture Translation")
    st.write("Enter text to translate it into SIBI sign language gestures.")
    
    text_input = st.text_input("Enter letters (e.g. HELLO)", value="HELLO").upper()
    
    if text_input:
        # Keep only alphabetical characters
        chars = [c for c in text_input if c.isalpha()]
        if chars:
            # Display images side-by-side in columns
            cols = st.columns(len(chars))
            for idx, char in enumerate(chars):
                img_path = f"static/images/{char}.jpg"
                if os.path.exists(img_path):
                    with cols[idx]:
                        st.image(img_path, caption=char, use_column_width=True)
                else:
                    with cols[idx]:
                        st.error(f"'{char}' not found")
        else:
            st.info("Please enter letters only.")

# ==========================================
# TAB 2: IMAGE UPLOAD
# ==========================================
with tab2:
    st.header("Image Upload Detection")
    st.write("Upload an image containing a SIBI sign language gesture for detection.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Run YOLO detection
        results = model(img_array)
        
        # Plot bounding box results (returns BGR format)
        annotated_img = results[0].plot()
        
        # Convert BGR to RGB for Streamlit
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        # Get prediction label
        pred = results[0].names[int(results[0].boxes.cls[0])] if results[0].boxes else "No detection"
        
        # Display comparison side-by-side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(annotated_img_rgb, caption="Detection Result", use_column_width=True)
            st.success(f"**Prediction:** {pred}")

# ==========================================
# TAB 3: WEBCAM SNAPSHOT
# ==========================================
with tab3:
    st.header("Webcam Detection")
    st.write("Take a photo using your webcam to detect SIBI sign language gestures.")
    
    img_file_buffer = st.camera_input("Take a picture")
    
    if img_file_buffer is not None:
        # Load image from camera buffer
        img = Image.open(img_file_buffer)
        img_array = np.array(img)
        
        # Run YOLO detection
        results = model(img_array)
        annotated_img = results[0].plot()
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        pred = results[0].names[int(results[0].boxes.cls[0])] if results[0].boxes else "No detection"
        
        # Display result
        st.image(annotated_img_rgb, caption="Webcam Detection Result", use_column_width=True)
        st.success(f"**Prediction:** {pred}")