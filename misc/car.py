import RPi.GPIO as GPIO
from time import sleep

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

ena = 14
enb = 17
mb2 = 4
mb1 = 3
ma2 = 18
ma1 = 15

GPIO.setup(ena, GPIO.OUT)
GPIO.setup(enb, GPIO.OUT)
GPIO.setup(mb1, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(mb2, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(ma1, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(ma2, GPIO.OUT, initial=GPIO.LOW)

pa = GPIO.PWM(ena,1000)
pa.start(50)
pb = GPIO.PWM(enb,1000)
pb.start(50)

while True:
    x = input()
    
    if x == 'f':
        #forward
        GPIO.output(ma2, GPIO.HIGH)
        GPIO.output(mb1, GPIO.HIGH)
        sleep(0.3)
        GPIO.output(ma2, GPIO.LOW)
        GPIO.output(mb1, GPIO.LOW)
    elif x == 'b':
        #back
        GPIO.output(ma1, GPIO.HIGH)
        GPIO.output(mb2, GPIO.HIGH)
        sleep(0.3)
        GPIO.output(ma1, GPIO.LOW)
        GPIO.output(mb2, GPIO.LOW)
    elif x == 'l':
        #left
        GPIO.output(mb1, GPIO.HIGH)
        sleep(0.3)
        GPIO.output(mb1, GPIO.LOW)
    elif x == 'r':
        #right
        GPIO.output(ma2, GPIO.HIGH)
        sleep(0.3)
        GPIO.output(ma2, GPIO.LOW)
    elif x == 'x':
        break
