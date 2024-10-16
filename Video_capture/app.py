from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import openai
import cv2
import os

# Initialize Flask app
app = Flask(__name__)

# Configure OpenAI API key
openai.api_key = "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"

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle speech recognition
@app.route('/recognize_speech', methods=['POST'])
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=text,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5
        )
        response_text = response.choices[0].text
        return jsonify({"text": text, "response": response_text})
    except Exception as e:
        return jsonify({"error": str(e)})

# Route to handle face detection
@app.route('/detect_faces', methods=['POST'])
def detect_faces():
    # Access the webcam
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    face_detected = False
    while not face_detected:
        ret, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            face_detected = True
            # (Optionally, you could save or process the frame here)

    cap.release()
    return jsonify({"message": "You are Human."})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
