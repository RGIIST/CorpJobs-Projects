from gtts import gTTS
import os

my_text="The sub-processes in the list above of course can differ, but these are roughly steps needed to approach automatic character recognition. In OCR software, it’s main aim to identify and capture all the unique words using different languages from written text characters."
language='en'
tts=gTTS(text=my_text,lang=language,slow=False, tld='co.in')
tts.save("ld.mp3")

# Loading the audio file to play
import audio_metadata
audio_metadata.load(".mp3")
