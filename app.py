import streamlit as st

# ✅ MUST be first Streamlit command
st.set_page_config(page_title="Emotion Detector", layout="centered")

import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import model_from_json

# Load the Haar Cascade Classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the model
@st.cache_resource
def load_emotion_model():
    with open("model_a.json", "r") as file:
        model_json = file.read()
    model = model_from_json(model_json)
    model.load_weights("model.weights.h5")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = load_emotion_model()
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def detect_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "No face detected", image

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        roi = cv2.resize(face, (48, 48))
        roi = roi[np.newaxis, :, :, np.newaxis]
        pred = model.predict(roi)
        emotion = EMOTIONS_LIST[np.argmax(pred)]

        # Draw rectangle and label
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return emotion, image

# UI
st.title("Emotion Detector")
st.write("Upload an image and detect the facial emotion.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Emotion"):
        emotion, result_img = detect_emotion(image)
        st.subheader(f"Detected Emotion: {emotion}")
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Result", use_column_width=True)
