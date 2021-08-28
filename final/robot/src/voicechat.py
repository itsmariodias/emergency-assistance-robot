import numpy as np
import tensorflow as tf
import random
import json
import time
import RPi.GPIO as GPIO
from time import sleep
 
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pyttsx3
import speech_recognition as sr
 
from ctypes import *
from contextlib import contextmanager

import phonecall
import medicinetray
from speaking import say, speak

from multiprocessing import Process

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
 
def py_error_handler(filename, line, function, err, fmt):
    pass
 
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

@contextmanager
def noalsaerr():
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)
    yield
    asound.snd_lib_error_set_handler(None)
        
def runInParallel(*fns):
    proc = []
    for fn in fns:
        p = Process(target=fn)
        p.start()
        proc.append(p)
    for p in proc:
        p.join()

def execute_response(response, action, phone, tray, name, number):
    if action == "yes_call" :
        call = phone.Call(phone=number, name=name)
        msg = "You have been contacted because you are the emergency contact for "+name+". Please hurry to their place of residence. They may be in danger. - EA Robot"
        if call == False:
            phone.SMS(phone=number, msg=msg)
    elif action in ["vaseline", "pain_cream"]:
        tray.OpenRightTray()
    elif action in ["bandaid", "stomach_pain", "ointment"]:
        tray.OpenLeftTray()
    elif action in ["thanks", "done"]:
        tray.CloseRightTray()
        tray.CloseLeftTray()
    else:
        pass


def VoiceChat(show_details = False, ERROR_THRESHOLD = 0.0, model_path="chatbotv4.tflite", data_path="intents.json", name="Test", number="9284547562"):

    phone = phonecall.PhoneCall()
    tray = medicinetray.Tray(20,21)

    #Open and preprocess the data in intents
    with open(data_path) as file:
        data = json.load(file)
 
    training_sentences = [] #x
    training_labels = [] #y

    for intent in data['intents']:
        for pattern in intent['patterns']:
            training_sentences.append(pattern)
            training_labels.append(intent['tag'])

    enc = LabelEncoder()
    enc.fit(training_labels)
    training_labels = enc.transform(training_labels)

    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>") # adding out of vocabulary token
    tokenizer.fit_on_texts(training_sentences)
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded = pad_sequences(sequences, truncating='post', maxlen=20)

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
         
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #Setup the text to speech engine.
    #engine = pyttsx3.init()
    #engine.setProperty('voice', 'english+f2')
    #voices = engine.getProperty('voices')
    #engine.setProperty('voice', voices[1].id)
    #volume = engine.getProperty('volume')
    #engine.setProperty('volume', 10.0)
    #rate = engine.getProperty('rate')
    #engine.setProperty('rate', rate - 50)
 
    context = ""
    action = ""
    
    t0 = time.time()
    count = 0
    silence = 0
    call_flag = False

    rec = sr.Recognizer()
    with noalsaerr():
        with sr.Microphone() as source:
            rec.adjust_for_ambient_noise(source, duration=1)
            
    response = 'Is there a problem?'
    print("Bot: ", response)
    say(response)
    timer = 300 #5min timer
    
    while True:
        
        try:
            response_flag = False
            with noalsaerr():
                                                  
                with sr.Microphone() as source:
                    #rec.adjust_for_ambient_noise(source)
                    
                    t1 = time.time()
                    if show_details: print("Time elapsed: ", t1 - t0)
                    if t1 - t0 > timer and call_flag == False:
                    
                        if timer == 15:
                            response = "Contacting emergency contacts."
                            timer = 300
                            context = ""
                        else:
                            response = "No response recorded for 5 mins, will attempt to call emergency contacts."
                            context = ""

                        print("Bot: ", response)
                        action = "yes_call"
                        runInParallel(say(response), execute_response(response, action, phone, tray, name, number))

                        response = 'Can I help you with anything else?'
                        print("Bot: ", response)
                        say(response)
                        call_flag = True

                    print("Speak...")

                    audio = rec.listen(source, timeout=4)
 
                    spoken = rec.recognize_google(audio, language='en-US')
 
                    #string = input('Enter: ')
                    print("You: ", spoken)

                    input_data = np.array(pad_sequences(tokenizer.texts_to_sequences([spoken]), truncating='post', maxlen=20),
                          dtype=np.float32)
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    result = interpreter.get_tensor(output_details[0]['index'])[0]

                    if show_details: print("result: ", result)
                    # filter out predictions below a threshold
                    results = [[i,r] for i,r in enumerate(result) if r>ERROR_THRESHOLD]
                    # sort by strength of probability
                    results.sort(key=lambda x: x[1], reverse=True)
 
                    return_list = []
                    for r in results:
                        return_list.append((enc.inverse_transform([r[0]])[0], r[1]))
                    if show_details: print("return_list: ", return_list)
 
                    if return_list:
                        for category,r in return_list:
                            if response_flag == False:
                                for i in data['intents']:
                                    if i['tag']==category:
                                        if context in i['context_filter']:
                                            response = np.random.choice(i['responses'])
                                            context = i["context_set"]
                                            action = category
                                            if show_details: print("current context: ", context)

                                            print("Bot: ", response)
                                            if action == 'yes_call':
                                                say(response)
                                                execute_response(response, action, phone, tray, name, number)
                                                call_flag = True
                                                response = 'Can I help you with anything else?'
                                                print("Bot: ", response)
                                                say(response)
                                                action = ""
                                            elif action in ['call', 'fall', 'critical']:
                                                say(response)
                                                timer = 15
                                            elif action == 'no_call':
                                                timer = 300
                                                say(response)
                                            else:
                                                execute_response(response, action, phone, tray, name, number)
                                                say(response)

                                            t0 = time.time()
                                            response_flag = True
 
                    if response_flag == False:
                        count += 1
                        error_sentence = "Sorry, can you repeat that?"
                        print("Bot: ", error_sentence)
                        say(error_sentence)
                        if count > 3:
                            print("Bot: ", response)
                            say(response)
                            count = 0
                    if response == 'Glad I could be of service! Goodbye.':
                        return
 
        except (sr.UnknownValueError, sr.WaitTimeoutError):
            silence += 1
            error_sentence = "Did you say something? I did not understand you."
            if silence > 5 and call_flag == False:
                silence = 0
                print("Bot: ", error_sentence)
                say(error_sentence)
                print("Bot: ", response)
                say(response)
        except sr.RequestError as e:
            message = 'My speech recognition service has failed. {0}'
            speak(message.format(e))
        except (KeyboardInterrupt, EOFError, SystemExit):
            # Press ctrl-c or ctrl-d on the keyboard to exit
            break
 
if __name__ == '__main__':
    VoiceChat(show_details = True, ERROR_THRESHOLD = 0.1, name="Mario")