
# Drive.AI

End-End framework for autonomous maneuvering of vehicles in Carla Simulator 
from point A to point B on a predefined trajectory following traffic signals 
and interacting appropriately with environment agents. 
### The task of maneuvering is divided into sub problems: 


- *PID controller* : Navigate the vehicle from point A to point B
- *Object Detection* : Detect Traffic signals and Vehicles
- *Traffic Signal Handling* : Overriding PID control as per Traffic Signals
- *Obstacle Detection* : Classify the detected objects are close with immediate impact, near with delayed impact and far with no impact on the trajectory
- *PID + Traffic Signals + Obstacles* : Combining all the models created.

### Basic Flow Diagram
![Basic Flow Diagram](https://github.com/shivanshu1641/Drive.AI/blob/main/Project%20Model.png?raw=true)

#### Language

[Python](https://linktodocumentation)

#### Model / Major Packages
[YoloV5](https://github.com/ultralytics/yolov5),
[OpenCV](https://opencv.org/),
[Pandas](https://pandas.pydata.org/),
[Pytorch](https://pytorch.org/)

#### Simulator
[Carla](https://carla.org/)



