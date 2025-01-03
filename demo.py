import streamlit as st
import numpy as np
import cv2
from keras.models import load_model


# Load the pre-trained model
@st.cache_resource
def load_trained_model():
    return load_model('age_detection_model.h5')


# Preprocess the uploaded or captured image
def preprocess_image(image_bytes):
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128))  # Resize to match the model input size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


def main():
    st.title("Age Detection Demo")
    st.write("Upload a face image or capture one using your camera to predict the age.")

    # Load the trained model
    model = load_trained_model()

    # Option to choose between uploading an image or capturing one
    option = st.radio("Choose an input method:", ["Upload an Image", "Use Device Camera"])

    image_bytes = None

    if option == "Upload an Image":
        uploaded_image = st.file_uploader("Upload an image...", type=['jpg', 'png', 'jpeg'])
        if uploaded_image is not None:
            image_bytes = uploaded_image.read()

    elif option == "Use Device Camera":
        captured_image = st.camera_input("Take a picture")
        if captured_image is not None:
            image_bytes = captured_image.read()

    if image_bytes:
        # Display the input image
        st.image(image_bytes, caption="Input Image", use_container_width=True)

        # Preprocess the image
        img = preprocess_image(image_bytes)

        # Predict the age
        predicted_age = model.predict(img)
        st.subheader(f"Predicted Age: {int(predicted_age[0][0])}")


if __name__ == '__main__':
    main()
