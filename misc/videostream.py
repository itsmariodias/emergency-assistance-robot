#!/usr/bin/env python

import cv2
import numpy as np
import tensorflow as tf
from threading import Thread

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        # ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stoppedq
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # Return the most recent frame
        return self.frame

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True

class ObjectDetect:

    def __init__(self, model_dir='model.tflite', labels_dir='flatlabels.txt'):

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

    def Scan(self, threshold=0.5, show_output=True, resolution="1280x720", display=True):

        floating_model = (self.input_details[0]['dtype'] == np.float32)
        input_mean = 127.5
        input_std = 127.5

        # Initialize frame rate calculation
        if display:
            frame_rate_calc = 1
            freq = cv2.getTickFrequency()

        imW, imH = map(int, resolution.split('x'))

        mid_x = imW / 2

        mid_y = imH / 2

        # Initialize video stream
        videostream = VideoStream(resolution=(imW,imH),framerate=30).start()

        try:
            while True:

                if display:
                    # Start timer (for calculating frame rate)
                    t1 = cv2.getTickCount()

                # Grab frame from video stream
                frame = videostream.read()

                # Acquire frame and resize to expected shape [1xHxWx3]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
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

                # Loop over all detections and draw detection box if confidence is above minimum threshold
                for i in range(len(scores)):
                    if ((scores[i] > threshold) and (scores[i] <= 1.0)):

                        # Get bounding box coordinates and draw box
                        # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                        ymin = int(max(1,(boxes[i][0] * imH)))
                        xmin = int(max(1,(boxes[i][1] * imW)))
                        ymax = int(min(imH,(boxes[i][2] * imH)))
                        xmax = int(min(imW,(boxes[i][3] * imW)))
                        
                        if display:
                            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                        
                            # Draw label
                            object_name = self.labels[int(classes[i])] # Look up object name from "labels" array using class index
                            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                            label_ymin = max(ymin, labelSize[1]-10) # Make sure not to draw label too close to top of window
                            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                        # Draw circle in center
                        xcenter = xmin + (int(round((xmax - xmin) / 2)))
                        ycenter = ymin + (int(round((ymax - ymin) / 2)))

                        if display:
                            cv2.circle(frame, (xcenter, ycenter), 5, (0,0,255), thickness=-1)

                        if (xcenter >= mid_x - 213 and xcenter < mid_x + 213):
                            location = "center"
                        elif xcenter <= mid_x:
                            location = "left"
                        else:
                        	location = "right"

                        # Print info
                        print(object_name+' at ('+str(xcenter)+', '+str(ycenter)+') : '+location)

                if display:
                    # Draw framerate in corner of frame
                    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

                    # All the results have been drawn on the frame, so it's time to display it.
                    cv2.imshow('Human Detection', frame)

                    # Calculate framerate
                    t2 = cv2.getTickCount()
                    time1 = (t2-t1)/freq
                    frame_rate_calc= 1/time1

                # Press 'q' to quit
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    videostream.stop()
                    break

        except (KeyboardInterrupt, EOFError, SystemExit):
            cv2.destroyAllWindows()
            videostream.stop()
            return

if __name__=="__main__":
    test = ObjectDetect(model_dir="/home/pi/catkin_ws/src/wall_follower_sim/src/detect.tflite", labels_dir="/home/pi/catkin_ws/src/wall_follower_sim/src/flatlabels.txt")
    test.Scan(resolution="640x480", display=True)
