import cv2
import pyttsx3
import face_recognition 
import speech_recognition as sr
import google.generativeai as genai

known_face = face_recognition.load_image_file("FelixJoseph.jpg")
known_face_encodings = face_recognition.face_encodings(known_face)[0]
known_face_name = "Felix Joseph"
genai.configure(api_key = "AIzaSyAXbyz-zY2CPOD8IvA1z8xkIfQJM4RWihg")
model = genai.GenerativeModel('gemini-1.5-flash')

def ask_question():
    #Initialize text to speech engine
    engine = pyttsx3.init()
    
    #Get the engine's voice property and set it to female voice, 0 for male voice
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    
    #Initialize speech recognition library
    recognizer = sr.Recognizer()
    
    #Get speech from your device microphone
    with sr.Microphone(device_index = 2) as source:
        #noise cancellation 
        recognizer.adjust_for_ambient_noise(source, duration = 0.5)
    
        #Record audio
        recorded_audio = recognizer.listen(source)
    
        try:
            #Convert the recorded audio to text
            text = recognizer.recognize_google(recorded_audio)
    
            #Pass text to Gemini
            response = model.generate_content(text)
    
            #Remove any * in the response text
            text = response.text.replace("*", "")
    
            #Convert the text back to audio
            engine.say(text)
    
            #Play the audio
            engine.runAndWait()
            
        except:
            text = "Sorry, i can't hear you"
            
            #Convert the text back to audio
            engine.say(text)
    
            #Play the audio
            engine.runAndWait()

name = "Unknown"

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        match = face_recognition.compare_faces([known_face_encodings], face_encoding)
        if True in match:
            name = known_face_name
        else:
            name = "Unknown"
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imshow("Facial Recognition", image)
    
    #Answer a question if face is recognized
    if name != "Unknown":
        ask_question()
        name = "Unknown"
        
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()