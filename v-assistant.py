import openai
import speech_recognition as sr
import pyttsx3
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('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')

openai.api_key = OPENAI_API_KEY

#function to convrt text to soeech
# speech 
def SpeakText(command):
    # initaioiz the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()

# initialze the recognizer
r = sr.Recognizer()

def record_text():
    # loop in case of errors
    while(1):
        try:
            # use the microphone as a source of input
            with sr.Microphone() as source2:
                # prepare recognizer to receive input
                r.adjust_for_ambient_noise(source2, duration = 0.2)
                print("I'm listening")
                # listen to the users input
                audio = r.listen(source2)
                # using google to recognize audio
                MyText = r.recognize_google(audio)
                return MyText
            
        except sr.RequestError as e:
            print("Could not request result; {0}.format(e)")

        except sr.UnknownValueError:
            print("Unknown error occurred")

def send_to_model():
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = [
            {"role": "user", "content": text}
        ]
    )

    message = response.choice[0].messages.content
    messages.append(response.choice[0].messages)
    return message

messages = []
while(1):
    text = record_text()
    messages.append({"role": "user", "content": text})
    response = send_to_model(messages)
    SpeakText(response)

    print(response)
    