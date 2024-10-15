import cv2
import pyttsx3
import face_recognition
import speech_recognition as sr
import google.generativeai as genai

# Load and encode the reference image of your face
known_face = face_recognition.load_image_file("felix.jpg")
known_face_encodings = face_recognition.face_encodings(known_face)[0]
known_face_name = "Felix Joseph"

# Configure the Generative AI model
genai.configure(api_key="AIzaSyAXbyz-zY2CPOD8IvA1z8xkIfQJM4RWihg")
model = genai.GenerativeModel('gemini-1.5-flash')

def ask_question():
    # Initialize text-to-speech engine
    engine = pyttsx3.init()
    
    # Set the engine's voice property to a female voice (index 1)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    
    # Initialize the speech recognition library
    recognizer = sr.Recognizer()
    
    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        
        print("Listening...")
        
        try:
            # Record audio with a timeout
            recorded_audio = recognizer.listen(source, timeout=10)
            
            # Convert the recorded audio to text
            text = recognizer.recognize_google(recorded_audio)
            print(f"Felix Joseph: {text}")
            
            # Pass the text to Gemini for generating a response
            response = model.generate_content(text)
            
            if response and response.text:
                # Remove any '*' in the response text
                response_text = response.text.replace("*", "")
                
                # Convert the response text to speech
                engine.say(response_text)
                engine.runAndWait()
            else:
                print("No response from model.")
                engine.say("Sorry, I didn't get a response.")
                engine.runAndWait()

        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            engine.say("Sorry, I could not understand the audio.")
            engine.runAndWait()

        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            engine.say("Sorry, I'm having trouble connecting to the speech recognition service.")
            engine.runAndWait()

        except Exception as e:
            print(f"Error during speech recognition or processing: {e}")
            engine.say("An error occurred. Please try again.")
            engine.runAndWait()

name = "Unknown"

# Initialize the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to capture image")
        break

    # Detect face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the detected face encoding with the known face encoding
        match = face_recognition.compare_faces([known_face_encodings], face_encoding)
        if True in match:
            name = known_face_name
        else:
            name = "Unknown"
        
        # Draw a rectangle around the face and label it
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # Display the video frame with face recognition
    cv2.imshow("Facial Recognition", image)
    
    # If the recognized face is yours, ask a question
    if name == known_face_name:
        ask_question()
        name = "Unknown"  # Reset the name after asking a question
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
