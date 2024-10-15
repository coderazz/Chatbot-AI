import tkinter as tk
from tkinter import messagebox
import speech_recognition as sr
import cv2
import openai

# Configure the OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Function to handle speech recognition
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        messagebox.showinfo("Info", "Please speak something...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        messagebox.showinfo("You said", text)

        try:
            # Get response from OpenAI
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=text,
                max_tokens=1024,
                n=1,
                stop=None,
                temperature=0.5
            )
            response_text = response.choices[0].text.strip()
            messagebox.showinfo("OpenAI's response", response_text)
        except Exception as e:
            messagebox.showerror("Error", f"Error generating text: {e}")

    except sr.UnknownValueError:
        messagebox.showerror("Error", "Sorry, could not understand the audio.")
    except sr.RequestError as e:
        messagebox.showerror("Error", f"Could not request results; {e}")

# Function to handle face recognition
def detect_faces():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image from camera.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Creating the main Tkinter window
root = tk.Tk()
root.title("Speech & Face Recognition")

# Adding buttons for the actions
btn_speech = tk.Button(root, text="Say Something", command=recognize_speech)
btn_speech.pack(pady=10)

btn_face = tk.Button(root, text="Detect Faces", command=detect_faces)
btn_face.pack(pady=10)

root.mainloop()