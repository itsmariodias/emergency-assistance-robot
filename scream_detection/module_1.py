import pyaudio
import RPi.GPIO as GPIO
import numpy as np
import params as yamnet_params
import yamnet as yamnet_model
from ctypes import *
from contextlib import contextmanager
import tensorflow as tf
from time import sleep

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

time_stamp = time.time()
direction_flag = 0
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

def direction_detect(channel):
    global time_stamp
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

    def __init__(self, left=19, right=26):
        direction_flag = 0 #default left

        GPIO.setup(6, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(13, GPIO.OUT, initial=GPIO.LOW)

        GPIO.setup(left, GPIO.IN)
        GPIO.setup(right, GPIO.IN)
        GPIO.add_event_detect(left, GPIO.FALLING, direction_detect, bouncetime=10)
        GPIO.add_event_detect(right, GPIO.FALLING, direction_detect, bouncetime=10)
 
        self.interpreter = tf.lite.Interpreter(model_path="yamnet.tflite")
        self.interpreter.allocate_tensors()
        self.inputs = interpreter.get_input_details()
        self.outputs = interpreter.get_output_details()
         
        self.params = yamnet_params.Params()
        self.yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')

    def listening(self):
        frame_len = 15600 # 0.975 sec
        cnt = 0
        with noalsaerr():
            p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=int(self.params.sample_rate),
                    input=True,
                    frames_per_buffer=frame_len)
        while True:
            try:
                # data read
                data = stream.read(frame_len, exception_on_overflow=False)
     
                # byte --> float
                frame_data = buf_to_float(data, n_bytes=2, dtype=np.int16)
     
                waveform = frame_data #/ 32768.0  # Convert to [-1.0, +1.0]
     
                # Convert to mono and the sample rate expected by YAMNet.
                if len(waveform.shape) > 1:
                    waveform = np.mean(waveform, axis=1)
     
                interpreter.set_tensor(self.inputs[0]['index'], np.expand_dims(np.array(waveform, dtype=np.float32), axis=0))
                interpreter.invoke()
                scores = interpreter.get_tensor(self.outputs[0]['index'])
     
                mean_scores = np.mean(scores, axis=0)
                top5_i = np.argsort(mean_scores)[::-1][:5]
     
                cnt+=1
     
                for i in top5_i:
                    if self.yamnet_classes[i] in ["Yell", "Screaming", "Crying, sobbing", 'Wail, moan', 'Groan']:
                        print("Scream detected.")
                        if direction_flag: print("Left")
                        else:  print("Right")
                        return True
                        # print("Relevant")
                        # print(yamnet_classes[i])
                        # sleep(2)
                        # break
                # else:
                #     print(cnt," Irrelevant")
                #     print(yamnet_classes[top5_i[0]])
     
            except (KeyboardInterrupt, EOFError, SystemExit):
                # Press ctrl-c or ctrl-d on the keyboard to exit
                stream.stop_stream()
                stream.close()
                p.terminate()
                break

        return False

#Start speaking into mic, if scream or pain-related sounds detected, it will say Relevant.