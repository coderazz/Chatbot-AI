from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import openai
import cv2
import os

# Initialize Flask app
app = Flask(__name__)

# Configure OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

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
        response_text = response.choices[0].text.strip()
        return jsonify({"text": text, "response": response_text})

    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand the audio."})
    except sr.RequestError as e:
        return jsonify({"error": f"Could not request results; {e}"})
    except Exception as e:
        return jsonify({"error": str(e)})

# Route to handle face detection
@app.route('/detect_faces', methods=['POST'])
def detect_faces():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({"error": "Failed to open webcam."}), 500

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_detected = False
    timeout = 10  # Timeout in seconds
    start_time = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            face_detected = True
            break

        # Check if the timeout has been reached
        elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        if elapsed_time > timeout:
            break

    cap.release()
    if face_detected:
        return jsonify({"Dominique": "Face detected. You are human."})
    else:
        return jsonify({"Dominique": "No face detected or timeout reached."}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
