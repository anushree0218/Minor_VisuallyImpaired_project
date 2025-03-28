# This script uses gTTS (Google Text-to-Speech) to convert text to speech and play it using pygame.
# It handles text chunking to ensure that the text does not exceed the character limit for gTTS.
# tts/text_to_speech.py
from gtts import gTTS
import pygame
import time
import os

class AudioAssistant:
    def __init__(self):
        pygame.mixer.init()
        self.temp_file = "temp_speech.mp3"
        
    def text_to_speech(self, text, lang='en'):
        try:
            # Generate speech
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(self.temp_file)
            
            # Play audio
            self._play_audio()
            
        except Exception as e:
            print(f"TTS Error: {str(e)}")
        finally:
            if os.path.exists(self.temp_file):
                os.remove(self.temp_file)
    
    def _play_audio(self):
        pygame.mixer.music.load(self.temp_file)
        pygame.mixer.music.play()
        
        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

# Usage example            
if __name__ == "__main__":
    assistant = AudioAssistant()
    
    with open("llm_output.txt", "r") as f:
        description = f.read()
        
    # Split into chunks under 500 characters for gTTS
    chunks = [description[i:i+500] for i in range(0, len(description), 500)]
    
    for chunk in chunks:
        assistant.text_to_speech(chunk)
        time.sleep(0.5)  # Pause between chunks