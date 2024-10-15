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

# Facial detection using OpenCV DNN module
def detect_faces(img_path):
    # Load the pre-trained model for face detection
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

    # Read the image
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Prepare the image for the model
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Detect faces
    net.setInput(blob)
    detections = net.forward()

    # Draw rectangles around detected faces
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Display the image with detected faces
    cv2.imshow("Detected Faces", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
# ask_question()
detect_faces('felix.jpg')

# Creating the main Tkinter window
root = tk.Tk()
root.title("Speech & Face Recognition")

# Adding buttons for the actions
btn_speech = tk.Button(root, text="Say Something", command=recognize_speech)
btn_speech.pack(pady=10)

btn_face = tk.Button(root, text="Detect Faces", command=detect_faces)
btn_face.pack(pady=10)

root.mainloop()