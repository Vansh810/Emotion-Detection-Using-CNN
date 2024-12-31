import streamlit as st
import numpy as np
import cv2
from keras.models import load_model


# Load model and labels
@st.cache_resource
def load_emotion_model():
    model = load_model("emotion_detection_model.h5")
    labels = ['angry', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    return model, labels


model, emotion_labels = load_emotion_model()

# Function to predict emotion from an image
def predict_emotion(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)   # Add batch dimension
    prediction = model.predict(img)
    emotion_index = np.argmax(prediction)
    emotion = emotion_labels[emotion_index]
    confidence = prediction[0][emotion_index]
    return emotion, confidence


# Streamlit UI
st.title("Emotion Detection")
st.write("Upload an image or capture one using your device camera to detect the emotion expressed.")

# Provide options to upload or capture an image
option = st.radio("Choose an input method:", ["Upload an Image", "Use Device Camera"])

image = None

if option == "Upload an Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        # Read and decode the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

elif option == "Use Device Camera":
    captured_file = st.camera_input("Take a picture")
    if captured_file:
        # Read and decode the captured image
        file_bytes = np.asarray(bytearray(captured_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if image is not None:
    # Display the input image
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Input Image", use_container_width=True)

    # Predict the emotion
    emotion, confidence = predict_emotion(image)
    st.write(f"**Emotion Detected:** {emotion}")
