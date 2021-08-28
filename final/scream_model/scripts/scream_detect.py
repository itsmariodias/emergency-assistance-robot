#!/usr/bin/env python3

import pyaudio
import RPi.GPIO as GPIO
import numpy as np
from ctypes import *
from contextlib import contextmanager
import tensorflow as tf
import time
import csv
import io

import rospy
from std_msgs.msg import Int16

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

time_stamp = time.time()
direction_flag = 0
left = 19
right = 26
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
 
def buf_to_float(x, n_bytes=2, dtype=np.float32):
    """Convert an integer buffer to floating point values.
    This is primarily useful when loading integer-valued wav data
    into numpy arrays.
    .. seealso:: :func:`librosa.util.buf_to_float`
    :parameters:
        - x : np.ndarray [dtype=int]
            The integer-valued data buffer
        - n_bytes : int [1, 2, 4]
            The number of bytes per sample in ``x``
        - dtype : numeric type
            The target output type (default: 32-bit float)
    :return:
        - x_float : np.ndarray [dtype=float]
            The input data buffer cast to floating point
    """
 
    # Invert the scale of the data
    scale = 1. / float(1 << ((8 * n_bytes) - 1))
 
    # Construct the format string
    fmt = '<i{:d}'.format(n_bytes)
 
    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)

# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_map_csv = io.StringIO(class_map_csv_text)
    class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
    class_names = class_names[1:]  # Skip CSV header
    return class_names

def direction_detect(channel):
    global time_stamp
    global direction_flag
    time_now = time.time()
    if (time_now - time_stamp) >= 0.3:
        if channel==left:
            GPIO.output(6, GPIO.HIGH)
            GPIO.output(13, GPIO.LOW)
            direction_flag = 0
            
        elif channel==right:
            GPIO.output(13, GPIO.HIGH)
            GPIO.output(6, GPIO.LOW)
            direction_flag = 1
    time_stamp = time_now

class ScreamModel:

    def __init__(self, left_pin=19, right_pin=26):
        global direction_flag
        direction_flag = 0 #default left
        global left
        left = left_pin
        global right
        right = right_pin

        GPIO.setup(6, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(13, GPIO.OUT, initial=GPIO.LOW)

        GPIO.setup(left, GPIO.IN)
        GPIO.setup(right, GPIO.IN)
        GPIO.add_event_detect(left, GPIO.FALLING, direction_detect, bouncetime=10)
        GPIO.add_event_detect(right, GPIO.FALLING, direction_detect, bouncetime=10)
 
        self.interpreter = tf.lite.Interpreter(model_path="/home/pi/catkin_ws/src/scream_model/scripts/yamnet.tflite")
        self.input_details = self.interpreter.get_input_details()
        self.waveform_input_index = self.input_details[0]['index']
        self.output_details = self.interpreter.get_output_details()
        self.scores_output_index = self.output_details[0]['index']
        self.embeddings_output_index = self.output_details[1]['index']
        self.spectrogram_output_index = self.output_details[2]['index']
         
        self.class_names = class_names_from_csv(open('/home/pi/catkin_ws/src/scream_model/scripts/yamnet_class_map.csv').read())
        
        self.pub = rospy.Publisher('scream_detect', Int16, queue_size=10)
        
        rospy.init_node('scream_detector', anonymous=True)
        
        self.rate = rospy.Rate(10)

    def listening(self):
        global direction_flag
        frame_len = 16000 # 1 sec
        cnt = 0
        with noalsaerr():
            self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=frame_len)

        print("Listening..")
        while True:
            try:
                # data read
                data = self.stream.read(frame_len, exception_on_overflow=False)
     
                # byte --> float
                waveform = buf_to_float(data, n_bytes=2, dtype=np.int16)
     
                # Convert to mono and the sample rate expected by YAMNet.
                if len(waveform.shape) > 1:
                    waveform = np.mean(waveform, axis=1)
     
                waveform = np.array(waveform, dtype=np.float32)

                self.interpreter.resize_tensor_input(self.waveform_input_index, [len(waveform)])
                self.interpreter.allocate_tensors()
                self.interpreter.set_tensor(self.waveform_input_index, waveform)
                self.interpreter.invoke()
                scores, embeddings, spectrogram = (
                    self.interpreter.get_tensor(self.scores_output_index),
                    self.interpreter.get_tensor(self.embeddings_output_index),
                    self.interpreter.get_tensor(self.spectrogram_output_index))
     
                mean_scores = np.mean(scores, axis=0)
                top5_i = np.argsort(mean_scores)[::-1][:5]
     
                cnt+=1
                
                #classes = []
     
                for i in top5_i:
                    #classes.append(self.class_names[i])
                    
                    if self.class_names[i] in ["Yell", "Screaming", "Crying, sobbing",
                                                  'Wail, moan', 'Groan', 'Burping, eructation', 'Baby cry, infant cry']:
                        #print(self.class_names[i])
                        if direction_flag == 0:
                            direction = -1
                            rospy.loginfo(direction)
                            self.pub.publish(direction)
                            self.rate.sleep()
                        else:
                            direction = 1
                            rospy.loginfo(direction)
                            self.pub.publish(direction)
                            self.rate.sleep()
                        return
                        # print("Relevant")
                        # print(yamnet_classes[i])
                        # sleep(2)
                        # break
                # else:
                #     print(cnt," Irrelevant")
                #     print(yamnet_classes[top5_i[0]])
                #print(cnt, classes)
     
            except (KeyboardInterrupt, EOFError, SystemExit):
                # Press ctrl-c or ctrl-d on the keyboard to exit
                self.stream.stop_stream()
                self.stream.close()
                self.p.terminate()
                break

        return 0

    def __del__(self):
        global left
        global right
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        GPIO.remove_event_detect(left)
        GPIO.remove_event_detect(right)
        GPIO.output(left, GPIO.LOW)
        GPIO.output(right, GPIO.LOW)

#Start speaking into mic, if scream or pain-related sounds detected, it will say Relevant.
        
if __name__ == '__main__':
    try:
        mod1 = ScreamModel()
        mod1.listening()
        
    except rospy.ROSInterruptException:
        pass