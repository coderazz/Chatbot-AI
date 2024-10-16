import speech_recognition as sr
import cv2
import face_recognition

# Load the reference image of your face and encode it
reference_img_path = 'FelixJoseph'  # Change this to the path of your reference image
reference_image = face_recognition.load_image_file(reference_img_path)
reference_encoding = face_recognition.face_encodings(reference_image)[0]

# Initialize the webcam
cap = cv2.VideoCapture(0)

recognizer = sr.Recognizer()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Convert the frame to RGB (face_recognition expects RGB images)
    rgb_frame = frame[:, :, ::-1]

    # Find all face encodings in the current frame of video
    face_encodings = face_recognition.face_encodings(rgb_frame)

    # Compare the detected face(s) with the reference encoding
    match = False
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces([reference_encoding], face_encoding)
        if True in matches:
            match = True
            print("Face recognized successfully!")

            # Recognize speech
            with sr.Microphone() as source:
                print("Speak something:")
                try:
                    audio = recognizer.listen(source, timeout=10)
                    text = recognizer.recognize_google(audio)
                    print("You said:", text)
                except sr.UnknownValueError:
                    print("Sorry, could not understand audio.")
                except sr.RequestError as e:
                    print(f"Could not request results from Google Speech Recognition service; {e}")
                except Exception as e:
                    print(f"Error during speech recognition: {e}")
            break  # Exit the loop after successful recognition

    if match:
        break

    # Display the resulting frame
    cv2.imshow('Live Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()