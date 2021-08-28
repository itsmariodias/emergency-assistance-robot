import numpy as np
import tensorflow as tf
import random
import json
 
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

class VoiceChat:
	
	def __init__(self):
