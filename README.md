# DeepAge

DeepAge is a real-time facial analysis application that combines age and emotion detection using deep learning models and computer vision. Built in Python, DeepAge leverages OpenCV for face detection, a pre-trained Caffe model for age estimation, and a Keras-based neural network for emotion recognition. The project features a Streamlit web interface for interactive webcam-based predictions.

## Features

- **Real-time Age Detection:** Utilizes a pre-trained deep neural network to estimate a person's age group from webcam input. Age groups include: (0-2), (4-6), (8-12), (15-20), (25-32), (38-43), (48-53), and (60-100).
- **Emotion Recognition:** Detects and classifies facial emotions such as Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral using a Keras model.
- **Web-based Interface:** Integrates Streamlit and streamlit-webrtc for live video streaming and interactive use.
- **Face Detection:** Employs OpenCV's Haar Cascade classifier for robust face localization on each video frame.

## How It Works

1. **Face Detection:** Each video frame from the webcam is processed to detect faces.
2. **Age Prediction:** The detected face region is preprocessed and passed to a deep neural network model, which outputs an age group.
3. **Emotion Prediction:** The same face region is also used for emotion prediction via a Keras model.
4. **Display:** Results are overlaid on the video stream in real-time, drawing bounding boxes and labels for both age and emotion.

## Requirements

- Python
- OpenCV
- Keras
- Streamlit
- streamlit-webrtc
- Pre-trained model files: `age_deploy.prototxt`, `age_net.caffemodel`, and `model.keras` (emotion model)

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shankar4112/DeepAge.git
   cd DeepAge
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Place model files:**  
   Download and place `age_deploy.prototxt`, `age_net.caffemodel`, and `model.keras` in the project directory.
4. **Run the Streamlit app:**
   ```bash
   streamlit run combine.py
   ```

## Usage

- The web application will start and access your webcam.
- It will display a real-time feed with bounding boxes around detected faces, showing the predicted age group and emotion for each face.
- To stop the application, close the web browser tab or press `q` if running from a script.

## Project Structure

- `combine.py`: Streamlit app integrating age and emotion detection for real-time webcam analysis.
- `age_model.py`: Standalone script for age prediction using OpenCV and the Caffe model.
- `README.md`: Project overview and setup instructions.

## License

This project currently does not specify a license. Please contact the repository owner for usage permissions.

---

For more details or to contribute, visit the [DeepAge GitHub repository](https://github.com/shankar4112/DeepAge).
