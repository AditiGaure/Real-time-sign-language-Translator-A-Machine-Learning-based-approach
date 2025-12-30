import cv2
import numpy as np
from keras.models import load_model
import streamlit as st
from PIL import Image

# Load the updated 26-class model
model = load_model("sign_language_model_full.h5")

# Define the class labels A-Z
class_labels = [chr(i) for i in range(65, 91)]  # A-Z

# Page Configuration
st.set_page_config(page_title="Real-Time Sign Language Translator: A Machine Learning Based Approach", layout="centered")
st.markdown("""
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #2c3e50;
        font-family: 'Arial', sans-serif;
    }
    .subtitle {
        font-size: 20px;
        color: #7f8c8d;
        font-family: 'Arial', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>Real-Time Sign Language Translator:  A Machine Learning Based Approach</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Model: <strong>Gesture-to-Text Translator (A-Z)</strong></div>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Start/Stop Camera Button
start = st.button(" Start Camera")
stop = st.button(" Stop Camera")

frame_placeholder = st.empty()
prediction_placeholder = st.empty()

if start:
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip image for mirror view
        frame = cv2.flip(frame, 1)

        # Define region of interest (ROI)
        x1, y1, x2, y2 = 100, 100, 300, 300
        roi = frame[y1:y2, x1:x2]

        # Preprocess the image
        roi_resized = cv2.resize(roi, (64, 64))
        roi_normalized = roi_resized / 255.0
        roi_reshaped = np.reshape(roi_normalized, (1, 64, 64, 3))

        # Predict
        prediction = model.predict(roi_reshaped)
        predicted_class = class_labels[np.argmax(prediction)]

        # Draw rectangle and prediction
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Predicted: {predicted_class}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2, cv2.LINE_AA)

        # Convert image for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)

        frame_placeholder.image(img_pil)
        prediction_placeholder.markdown(
            f"### 🔡 Predicted Output: <span style='color:blue;font-size:28px'>{predicted_class}</span>",
            unsafe_allow_html=True)

        # Stop condition
        if stop:
            break

    cap.release()
    cv2.destroyAllWindows()
