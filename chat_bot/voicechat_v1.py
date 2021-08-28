import numpy as np
import tensorflow as tf
import random
import json

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pyttsx3
import speech_recognition as sr
import winsound

from sklearn.preprocessing import LabelEncoder

with open("intents.json") as file:
    data = json.load(file)

training_sentences = [] #x
training_labels = [] #y
tags = [] #classes
responses = []


for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent['responses'])
    
    if intent['tag'] not in tags:
        tags.append(intent['tag'])



enc = LabelEncoder()
enc.fit(training_labels)
training_labels = enc.transform(training_labels)

vocab_size = 10000
embedding_dim = 16
max_len = 20
trunc_type = 'post'
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token) # adding out of vocabulary token
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, truncating=trunc_type, maxlen=max_len)

model = keras.models.load_model("chat_bot_final")

def VoiceChat(show_details = False, ERROR_THRESHOLD = 0.0):

    engine = pyttsx3.init(driverName='sapi5')
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    volume = engine.getProperty('volume')
    engine.setProperty('volume', 10.0)
    rate = engine.getProperty('rate')
    #engine.setProperty('rate', rate - 25)

    context = ""

    print('Bot: Distress detected. Is there a problem?')
    engine.say('Distress detected. Is there a problem?')
    engine.runAndWait()

    rec = sr.Recognizer()
    while True:  
        try:
            response_flag = False
            with sr.Microphone() as source:
                if show_details: print("Speak...")
                winsound.Beep(440, 250)
                
                audio = rec.listen(source)
                
                spoken = rec.recognize_google(audio, language='en-US')

                #string = input('Enter: ')
                print("You: ", spoken)
                result = model.predict(pad_sequences(tokenizer.texts_to_sequences([spoken]),
                                                     truncating=trunc_type, maxlen=max_len))[0]
                print(result)
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
                                        print("Bot: ", response)
                                        engine.say(response)
                                        engine.runAndWait()
                                        context = i["context_set"]
                                        if show_details: print("current context: ", context)
                                        response_flag = True

                if response_flag == False:
                    response = np.random.choice(["Sorry, I didn't understand that.", "Sorry, can you repeat that?", 
                                            "I did not understand what you said. Please repeat it."])
                    print("Bot: ", response)
                    engine.say(response)
                    engine.runAndWait()

                if response == 'Glad I could be of service! Goodbye.':
                    return

        except sr.UnknownValueError:
            continue
        except sr.RequestError as e:
            message = 'My speech recognition service has failed. {0}'
            engine.say(message.format(e))
        except (KeyboardInterrupt, EOFError, SystemExit):
            # Press ctrl-c or ctrl-d on the keyboard to exit
            break

VoiceChat(show_details = False, ERROR_THRESHOLD = 0.01)