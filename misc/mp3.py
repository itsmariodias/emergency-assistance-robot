from gtts import gTTS
import re
import pygame
import time
import sys

pygame.init()

text = "Unable to find source of distress. Please report."
# tts = gTTS(text=text, lang='en')
# tts.save(re.sub('[^A-Za-z0-9]+', '', text)+'.mp3')
def say(text):
    file = re.sub('[^A-Za-z0-9]+', '', text)+'.mp3'
    pygame.mixer.init()
    pygame.mixer.music.load(file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

say(text)

# Unable to find source of distress. Please report.

# Is there a problem?

# Contacting emergency contacts.

# No response recorded for 5 mins, will attempt to call emergency contacts.

# Can I help you with anything else?

# Sorry, can you repeat that?

# Glad I could be of service! Goodbye.

# Did you say something? I did not understand you.
