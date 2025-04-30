import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from keras.models import load_model

# Load pre-trained models (Make sure these files are available in the correct path)
emotion_model = load_model("model.keras")
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define a function for processing each frame
class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convert the stream frame to ndarray for OpenCV processing
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for face detection
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)  # Detect faces in the frame

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]  # Get the region of interest for the face
            face_roi = cv2.resize(face_roi, (48, 48))  # Resize face for emotion model input
            face_roi = face_roi / 255.0  # Normalize pixel values for emotion model
            face_roi = np.expand_dims(face_roi, axis=-1)  # Add channel dimension
            face_roi = np.expand_dims(face_roi, axis=0)   # Add batch dimension

            # Predict emotion
            emotion_pred = emotion_model.predict(face_roi, verbose=0)
            emotion_label = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'][np.argmax(emotion_pred)]

            # Predict age using DNN model
            face_blob = cv2.dnn.blobFromImage(img[y:y+h, x:x+w], 1.0, (227, 227),
                                              (78.426, 87.768, 114.895), swapRB=False)
            age_net.setInput(face_blob)
            age_preds = age_net.forward()
            age_label = AGE_BUCKETS[age_preds[0].argmax()]

            # Add the results to the frame (emotion and age)
            text = f'{emotion_label}, {age_label}'
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw rectangle around the face
            cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)  # Add text for predictions

        return img  # Return the modified frame

# Streamlit frontend
st.title("Real-Time Emotion and Age Detection")

# Start WebRTC stream and link to the VideoProcessor class for real-time video processing
webrtc_streamer(key="camera", video_processor_factory=VideoProcessor)
