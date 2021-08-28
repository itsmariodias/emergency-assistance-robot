import serial
import os, time
import RPi.GPIO as GPIO

from speaking import say, speak

class PhoneCall:
    
    def __init__(self):
        self.port = serial.Serial("/dev/ttyS0", baudrate=9600, timeout=1)
        self.port.write(b'AT\r')
        rcv = self.port.read(10)
        print(rcv.decode('UTF-8'))

    def Call(self, phone="9284547562", name="Test"):
        call_active = False
        cnt = 1
        while call_active == False and cnt <= 3:
            print("Attempting call..",cnt)
            cnt += 1
            call_string = 'ATD'+phone+';\r'
            self.port.write(call_string.encode('UTF-8'))
            t0 = time.time()
            t1 = time.time()
            while t1 - t0 < 60:
                if "MO CONNECTED" in self.port.read(30).decode('UTF-8'):
                    call_active = True
                    break
                t1 = time.time()
            
            if call_active:
                t0 = time.time()
                t1 = time.time()
                while t1 - t0 < 60:
                    speak("You have been contacted because you are the emergency contact for "+name+". Please hurry to their place of residence. They may be in danger.")
                    t1 = time.time()
                    if "NO CARRIER" in self.port.read(30).decode('UTF-8'):
                        break

        self.port.write(b'ATH\r')
        print("Call ended.")
        return call_active

    def SMS(self, phone="9284547562", msg="This is a test message"):
        self.port.write(b"AT+CMGF=1\r")
        time.sleep(3)
        sms_string = 'AT+CMGS="'+phone+'"\r'
        self.port.write(sms_string.encode('UTF-8'))
        print("Sending Message.")
        time.sleep(3)
        self.port.reset_output_buffer()
        time.sleep(1)
        self.port.write(str.encode(msg+chr(26)))
        time.sleep(3)
        print(self.port.read(60).decode('UTF-8'))
        print("Message Sent.")