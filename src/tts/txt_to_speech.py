from gtts import gTTS
import os

class TextToSpeech:
    def __init__(self, voice="en-US-Wavenet-D", speed=1.0):
        self.voice = voice
        self.speed = speed

    def speak(self, text):
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("output.mp3")
        os.system("mpg123 output.mp3")  # Ensure mpg123 is installed