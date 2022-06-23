"""
Copyright 2022 Mario Dias, Hansie Aloj, Nijo Ninan, Dipali Koshti.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

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