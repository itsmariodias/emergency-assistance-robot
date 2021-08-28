import numpy as np
import tensorflow as tf
import random
import json
import time
 
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
 
import pyttsx3
import speech_recognition as sr
 
from ctypes import *
from contextlib import contextmanager

import phonecall
import medicinetray

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

def execute_response(response, action, engine, phone, tray, name):
    if action == "yes_call" :
        phone.Call(phone="9284547562", name)
        phone.SMS(phone="9284547562", msg="This is a test message")

    elif action in ["yes_vaseline", "yes_pain_cream"]:
        tray.OpenRightTray()
    elif action in ["yes_bandaid", "yes_antacid", "yes_lotion"]:
        tray.OpenLeftTray()
    elif action in ["thanks", "done"]:
        tray.CloseRightTray()
        tray.CloseLeftTray()
    else:
        pass


def VoiceChat(show_details = False, ERROR_THRESHOLD = 0.0, model_path="chatbot.tflite", data_path="intents.json", name):

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
    engine = pyttsx3.init()
    engine.setProperty('voice', 'english+f3')
    #voices = engine.getProperty('voices')
    #engine.setProperty('voice', voices[1].id)
    #volume = engine.getProperty('volume')
    #engine.setProperty('volume', 10.0)
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate - 25)
 
    context = ""
    action = ""
 
    response = 'Bot: Distress detected. Is there a problem?'

    t0 = time.time()
    call_flag = False

    rec = sr.Recognizer()
    while True:

        t1 = time.time() 
        if t1 - t0 == 300 and call_flag == False:

            response = "No response recorded for 5 mins, will attempt to call emergency contacts."
            action = "yes_call"
            call_flag = True
            return

        try:
            response_flag = False
            with noalsaerr():
                with sr.Microphone() as source:
                    #winsound.Beep(440, 250)
                    rec.adjust_for_ambient_noise(source)

                    print("Bot: ", response)
                    engine.say(response)
                    engine.runAndWait()

                    # Execute based on response
                    execute_response(response, action, engine, phone, tray, name)

                    if show_details: print("Speak...")

                    audio = rec.listen(source)
 
                    spoken = rec.recognize_google(audio, language='en-US')
 
                    #string = input('Enter: ')
                    print("You: ", spoken)

                    input_data = np.array(pad_sequences(tokenizer.texts_to_sequences([spoken]), truncating=trunc_type, maxlen=max_len),
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

                                            if action == "yes_call":
                                                call_flag = True

                                            t0 = time.time()
                                            response_flag = True
 
                    if response_flag == False:
                        response = np.random.choice(["Sorry, I didn't understand that.", "Sorry, can you repeat that?", 
                                                "I did not understand what you said. Please repeat it."])
                    if response == 'Glad I could be of service! Goodbye.':
                        return
 
        except sr.UnknownValueError:
            response = np.random.choice(["Sorry, I didn't understand that.", "Sorry, can you repeat that?", 
                                                "I did not understand what you said. Please repeat it."])
        except sr.RequestError as e:
            message = 'My speech recognition service has failed. {0}'
            engine.say(message.format(e))
        except (KeyboardInterrupt, EOFError, SystemExit):
            # Press ctrl-c or ctrl-d on the keyboard to exit
            break
 
if __name__ == '__main__':
    VoiceChat(show_details = True, ERROR_THRESHOLD = 0.01, name="Test")