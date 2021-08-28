#!/usr/bin/env python3
import rospy # Python library for ROS
from sensor_msgs.msg import LaserScan # LaserScan type message is defined in sensor_msgs
from std_msgs.msg import Int16

from geometry_msgs.msg import Twist

from speaking import say

import cv2
import numpy as np
import tensorflow as tf
from threading import Thread
from videostream import VideoStream
import time

from voicechat import VoiceChat

class Navigation:

    def __init__(self, model_dir='model.tflite', labels_dir='flatlabels.txt'):

        self.setup_flag = True
        self.center = self.left = self.right = self.interval = 0
        self.factor = 0.01
        self.detect_flag = False
        self.xcenter = 0
        self.ycenter = 0

        # Laser scan range threshold
        self.thr1 = 0.5
        self.thr2 = 0.5
        
        #bot speeds
        self.lin_speed = 0.3
        self.ang_speed = 1.2
        self.detect_speed = 1.1
        
        self.direction_flag = 0
        self.keep_flag = False
        self.reverse_flag = False
        self.movee_flag = True
        self.endstream = False
        
        self.location = ""
        
        self.t0 = time.time()
        self.t1 = time.time()

        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)  # Publisher object which will publish "Twist" type messages
                                                 # on the "/cmd_vel" Topic, "queue_size" is the size of the
                                                             # outgoing message queue used for asynchronous publishing
                                                             
        rospy.Subscriber("scream_detect", Int16, self.screamdetect)

        self.move = Twist() # Creates a Twist message type object

        with open(labels_dir, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        if self.labels[0] == '???':
            del(self.labels[0])

        self.interpreter = tf.lite.Interpreter(model_path=model_dir)
        self.interpreter.allocate_tensors()

        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        
    def lasernav(self, dt):

        if self.setup_flag:
            self.factor = len(dt.ranges)*1.0/360
            self.center = int(len(dt.ranges)/2) + 1
            self.left = self.center + int(60*self.factor)
            self.right = self.center + int(-60*self.factor)
            
            self.interval = int(15*self.factor)
            self.interval_center = int(10*self.factor)

            self.setup_flag = False
        
        center_range = np.ma.masked_equal(dt.ranges[self.center-self.interval_center:self.center+self.interval_center], 0.0, copy=False).min()
        left_range = np.ma.masked_equal(dt.ranges[self.left-self.interval:self.left+self.interval], 0.0, copy=False).min()
        right_range = np.ma.masked_equal(dt.ranges[self.right-self.interval:self.right+self.interval], 0.0, copy=False).min()
        if self.direction_flag:
            if time.time() - self.t1 > 600:
                self.move.linear.x = 0.0
                self.move.angular.z = 0.0
                self.pub.publish(self.move)
                print("Unable to find source of distress. Please report.")
                say("Unable to find source of distress. Please report.")
                self.t1 = time.time()
            if time.time() - self.t0 > 1200:
                self.move.linear.x = 0.0
                self.move.angular.z = 0.0
                self.pub.publish(self.move)
                self.endstream = True
                self.nav.unregister()
            
            if self.detect_flag: 
                if(self.xcenter >= self.mid_x - 60 and self.xcenter < self.mid_x + 60):
                    self.move.linear.x = 0.3
                    self.move.angular.z = 0.0
                    self.location = "center"
                elif self.xcenter <= self.mid_x and self.xcenter != 0:
                    self.move.linear.x = 0.0
                    self.move.angular.z = self.detect_speed
                    self.location = "left"
                elif self.xcenter >= self.mid_x and self.xcenter != 0:
                    self.move.linear.x = 0.0
                    self.move.angular.z = -self.detect_speed
                    self.location = "right"
            
                if self.xcenter == 0 and self.location == "left":
                    self.move.linear.x = 0.0
                    self.move.angular.z = -self.detect_speed
                    self.location == "right"
                elif self.xcenter == 0 and self.location == "right":
                    self.move.linear.x = 0.0
                    self.move.angular.z = self.detect_speed
                    self.location == "left"   
                if center_range < 0.5 and self.location=="center":
                    self.move.linear.x = 0.0
                    self.move.angular.z = 0.0
                    self.pub.publish(self.move)
                    self.endstream = True
                    self.nav.unregister()
                    return
                    
                self.pub.publish(self.move) # publish the move object
                print(self.location)
            else:
                if self.direction_flag == -1:
                    if left_range > self.thr2 and not self.keep_flag:
                        self.move.linear.x = 0.0
                        self.move.angular.z = self.ang_speed
                        self.reverse_flag = False
                    elif center_range > self.thr1 and left_range > 0.2 and right_range > 0.2 and not self.reverse_flag:
                        self.move.linear.x = self.lin_speed
                        self.move.angular.z = 0.0
                        self.keep_flag = False
                    elif right_range > self.thr2:
                        self.move.linear.x = 0.0
                        self.move.angular.z = -self.ang_speed
                        self.keep_flag = True
                        self.reverse_flag = False
                    elif left_range <= 0.2 or center_range <= 0.2 or right_range <= 0.2:
                        self.move.linear.x = -self.lin_speed
                        self.move.angular.z = 0.0
                        self.keep_flag = False
                        self.reverse_flag = True
                elif self.direction_flag == 1:
                    if right_range > self.thr2 and not self.keep_flag:
                        self.move.linear.x = 0.0
                        self.move.angular.z = -self.ang_speed
                        self.reverse_flag = False
                    elif center_range > self.thr1 and left_range > 0.2 and right_range > 0.2 and not self.reverse_flag:
                        self.move.linear.x = self.lin_speed
                        self.move.angular.z = 0.0
                        self.keep_flag = False
                    elif left_range > self.thr2:
                        self.move.linear.x = 0.0
                        self.move.angular.z = self.ang_speed
                        self.keep_flag = True
                        self.reverse_flag = False
                    elif left_range <= 0.2 or center_range <= 0.2 or right_range <= 0.2:
                        self.move.linear.x = -self.lin_speed
                        self.move.angular.z = 0.0
                        self.keep_flag = False
                        self.reverse_flag = True
                self.pub.publish(self.move) # publish the move object
        else:
            if self.detect_flag: 
                if(self.xcenter >= self.mid_x - self.mid_x/3 and self.xcenter < self.mid_x + self.mid_x/3):
                    if self.location == "center":
                        self.t1 = time.time()
                    else:
                        self.location = "center"
                        self.t0 = time.time()
                        self.t1 = time.time()  
                elif self.xcenter <= self.mid_x and self.xcenter != 0:
                    if self.location == "left":
                        self.t1 = time.time()
                    else:
                        self.location = "left"
                        self.t0 = time.time()
                        self.t1 = time.time() 
                elif self.xcenter >= self.mid_x and self.xcenter != 0:
                    if self.location == "right":
                        self.t1 = time.time()
                    else:
                        self.location = "right"
                        self.t0 = time.time()
                        self.t1 = time.time()
                print(self.location)
                        
                if self.t1 - self.t0 >= 600:
                    if self.location == "right":
                        self.direction_flag = 1
                    else:
                        self.direction_flag = -1
                    
            else:
                self.t1 = time.time()
                if self.t1 - self.t0 >= 600:
                    if self.location == "right":
                        self.direction_flag = 1
                    else:
                        self.direction_flag = -1
            

    def screamdetect(self, direction):
        print("Scream Heard")
        self.direction_flag = direction.data

    def detection(self, threshold=0.5, resolution="640x480", display=False):

        floating_model = (self.input_details[0]['dtype'] == np.float32)
        input_mean = 127.5
        input_std = 127.5

        # Initialize frame rate calculation
        if display:
            frame_rate_calc = 1
            freq = cv2.getTickFrequency()

        imW, imH = map(int, resolution.split('x'))

        self.mid_x = imW / 2

        self.mid_y = imH / 2

        # Initialize video stream
        videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
        
        self.nav = rospy.Subscriber("/scan", LaserScan, self.lasernav)  # Subscriber object which will listen "LaserScan" type messages
                                                          # from the "/scan" Topic and call the "callback" function
                                  # each time it reads something from the Topic
        
        t = 0

        try:
            while True:
                if display:
                    # Start timer (for calculating frame rate)
                    t1 = cv2.getTickCount()

                # Grab frame from video stream
                frame = videostream.read()

                # Acquire frame and resize to expected shape [1xHxWx3]
                #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame, (self.width, self.height))
                input_data = np.expand_dims(frame_resized, axis=0)

                # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
                if floating_model:
                    input_data = (np.float32(input_data) - input_mean) / input_std

                # Perform the actual detection by running the model with the image as input
                self.interpreter.set_tensor(self.input_details[0]['index'],input_data)
                self.interpreter.invoke()

                # Retrieve detection results
                boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0] # Bounding box coordinates of detected objects
                classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0] # Class index of detected objects
                scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0] # Confidence of detected objects
                # num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

                indices = np.argwhere(classes == 0)
                boxes = np.squeeze(boxes[indices], axis=1)[0:1]
                scores = np.squeeze(scores[indices], axis=1)[0:1]
                classes = np.squeeze(classes[indices], axis=1)[0:1]

                self.xcenter = 0
                self.ycenter = 0
                # Loop over all detections and draw detection box if confidence is above minimum threshold
                for i in range(len(scores)):
                    if ((scores[i] > threshold) and (scores[i] <= 1.0)):

                        # Get bounding box coordinates and draw box
                        # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                        ymin = int(max(1,(boxes[i][0] * imH)))
                        xmin = int(max(1,(boxes[i][1] * imW)))
                        ymax = int(min(imH,(boxes[i][2] * imH)))
                        xmax = int(min(imW,(boxes[i][3] * imW)))
                        
                        # Draw circle in center
                        self.xcenter = xmin + (int(round((xmax - xmin) / 2)))
                        self.ycenter = ymin + (int(round((ymax - ymin) / 2)))
                            
                        if display:
                            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                            
                            # Draw label
                            object_name = self.labels[int(classes[i])] # Look up object name from "labels" array using class index
                            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                            label_ymin = max(ymin, labelSize[1]-10) # Make sure not to draw label too close to top of window
                            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                            #Put dot at center and print info
                            cv2.circle(frame, (self.xcenter, self.ycenter), 5, (0,0,255), thickness=-1)
                            print(object_name+' at ('+str(self.xcenter)+', '+str(self.ycenter)+') : ',int(scores[i]*100))
                            
                #if detected set detect flag to true, if last detected was 5 sec ago ignore false postives
                if self.xcenter and self.ycenter:
                    self.detect_flag = True
                    t = time.time()
                elif time.time() - t > 10:
                    self.detect_flag = False
                    

                if display:
                    # Draw framerate in corner of frame
                    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

                    # All the results have been drawn on the frame, so it's time to display it.
                    cv2.imshow('Human Detection', frame)

                    # Calculate framerate
                    t2 = cv2.getTickCount()
                    time1 = (t2-t1)/freq
                    frame_rate_calc= 1/time1
                    
                    
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    videostream.stop()
                    break

                if self.endstream:
                    cv2.destroyAllWindows()
                    videostream.stop()
                    break
                
        except KeyboardInterrupt or SystemExit:
            cv2.destroyAllWindows()
            videostream.stop()
            return

if __name__=="__main__":
    try:
        rospy.init_node('obstacle_avoidance_node') # Initializes a node
        robot = Navigation(model_dir="/home/pi/catkin_ws/src/robot/src/detect.tflite",
                           labels_dir="/home/pi/catkin_ws/src/robot/src/flatlabels.txt")
        robot.detection(display=True, resolution="640x360", threshold=0.6)
        Mod3 = VoiceChat(model_path="/home/pi/catkin_ws/src/robot/src/chatbotv4.tflite",
                         data_path="/home/pi/catkin_ws/src/robot/src/intents.json",
                         show_details = True, ERROR_THRESHOLD = 0.1, name="Mario")
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
