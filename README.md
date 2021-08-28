# First Aid and Emergency Assistance Robot using Deep Learning
Code base for my third-year minor project at Fr. Conceicao Rodrigues College of Engineering.

### Abstract:
Thanks to urbanization and societal changes, there has been an increase in the number of people living alone. This can be concerning for the elderly as many mishaps or accidents can happen. We propose a smart IOT based robot system to assist and help people, especially elderly, in case they are injured or in a state of emergency and when no one else is present to assist them in such a tragedy. The robot should be able to detect anomalies and provide first aid to the victim or call emergency contacts if necessary. We divide the overall functioning into three stages: distress detection, navigation and searching and assistance and feedback. The robot will be able to detect distress in forms of audible screams and it will also monitor the surroundings frequently. Once the robot predicts that a situation has occurred it will try to detect the person in its frame while navigating throughout the household. After successfully detecting the person, the robot will move towards them and begin its communication stage. The robot will attempt to get feedback from the person and try to provide appropriate remedy accordingly. If the victim is unconscious then it will contact emergency services immediately. The prototypes were designed and tested with different test cases to draw conclusions and evaluate limitations and scope for future improvements.

### Requirements:
The codebase was developed and run on a Raspberry Pi 4 Model B on Raspbian Buster. Requires ROS Melodic.  
Tutorial to install can be found [here](https://www.instructables.com/ROS-Melodic-on-Raspberry-Pi-4-RPLIDAR/).

#### Python Libraries:
```
numpy
tensorflow
sklearn
pyttsx3
speech-recognition
RPi.GPIO
rospy
opencv-python
gtts
pygame
```

#### Hardware Used:
```
Raspberry Pi 4 Model B - 4GB RAM w/ 33GB SD Card
Arduino Nano
YDLIDAR X2L
SIM900A GSM Module
Logitech C270 HD Webcam
L298N Motor Driver
LM393 Sound Detection Sensor Module
SG90 Servo Motor
```

### Implementation:
The `final` folder contains the folders/files that will go into the `src` folder of your ROS workspace. You need to build the new folders you add before performing any process. To launch the entire process you simply need to use the command `roslaunch robot robot.py`. However some knowledge of ROS is required to edit the launch files depending on the hardware being used. Also packages will need to installed into ROS where success may vary on the ROS and Raspbian versions you use. I faced difficulty in installing some packages related to `sensor_msgs` and `nav_msgs`.

### Images:
![robot](images/robot.png?raw=true)

![component_diagram](images/component_diagram.png?raw=true)

### References:
* [L. Wu, J. Lu, T. Zhang and J. Gong, "Robot-assisted intelligent emergency system for individual elderly independent living," 2016 IEEE Global Humanitarian Technology
Conference (GHTC), Seattle, WA, 2016, pp. 628-633, doi:10.1109/GHTC.2016.7857344. ](https://ieeexplore.ieee.org/document/7857344)  
* [Do, H.M., Sheng, W. & Liu, M. Human-assisted sound event recognition for home service robots. Robot. Biomim. 3, 7 (2016). https://doi.org/10.1186/s40638-016-0042-2](https://jrobio.springeropen.com/articles/10.1186/s40638-016-0042-2)  
* [Tensorflow Model Garden](https://github.com/tensorflow/models)  
* ["Speed/accuracy trade-offs for modern convolutional object detectors." Huang J, Rathod V, Sun C, Zhu M, Korattikara A, Fathi A, Fischer I, Wojna Z, Song Y, Guadarrama S, Murphy K, CVPR 2017](https://arxiv.org/abs/1611.10012)  
* [Shawn Hershey, Sourish Chaudhuri, Daniel P. W. Ellis, Jort F. Gemmeke, Aren Jansen, Channing Moore, Manoj Plakal, Devin Platt, Rif A. Saurous, Bryan Seybold, Malcolm Slaney, Ron Weiss and Kevin Wilson, “CNN Architectures for Large-Scale Audio Classification,” International Conference on Acoustics, Speech and Signal Processing (ICASSP), IEEE (2017).](https://research.google/pubs/pub45611/)
* [Zhang, A. (2017). Speech Recognition (Version 3.8) [Software]](https://pypi.org/project/SpeechRecognition/)  
* [Tensorflow Lite Object Detection Tutorial](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi)
