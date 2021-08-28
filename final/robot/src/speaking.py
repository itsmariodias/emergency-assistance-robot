from gtts import gTTS
import pygame
from io import BytesIO
import re

pygame.init()
    
def say(text):
    file = re.sub('[^A-Za-z0-9]+', '', text)+'.mp3'
    pygame.mixer.init()
    pygame.mixer.music.load('/home/pi/catkin_ws/src/robot/src/voices/'+file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
        
def speak(text):
    tts = gTTS(text=text, lang='en')
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    pygame.mixer.init()
    pygame.mixer.music.load(fp)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)