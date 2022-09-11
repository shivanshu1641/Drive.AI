## PID Controller Results
* [PID Conroller 101](https://github.com/shivanshu1641/Drive.AI/tree/main/Explaination%20%2B%20Results/1.%20PID%20Controller#pid-controller-101)
* [PID Controller In CARLA Simulator](https://github.com/shivanshu1641/Drive.AI/tree/main/Explaination%20%2B%20Results/1.%20PID%20Controller#pid-controller-in-carla-simulator)
* [Output Snippets](https://github.com/shivanshu1641/Drive.AI/tree/main/Explaination%20%2B%20Results/1.%20PID%20Controller#output-snippets)
* [Position Graph](https://github.com/shivanshu1641/Drive.AI/tree/main/Explaination%20%2B%20Results/1.%20PID%20Controller#position-graph)
* [Speed Graph](https://github.com/shivanshu1641/Drive.AI/tree/main/Explaination%20%2B%20Results/1.%20PID%20Controller#speed-graph)
### PID Controller 101
![PID Controller](https://github.com/shivanshu1641/Drive.AI/blob/main/Explaination%20+%20Results/1.%20PID%20Controller/FlowChart1.png?raw=true)

### PID Controller In CARLA Simulator 
![PID Controller In Carla Simulator](https://github.com/shivanshu1641/Drive.AI/blob/main/Explaination%20+%20Results/1.%20PID%20Controller/FlowChart2.png?raw=true)

### Output Snippets
* Green Dotted Path is the path, vehicle is *supposed* to follow.
* Red Dotted Path is the path, vehicle is *currently* following.
![Image1](https://github.com/shivanshu1641/Drive.AI/blob/main/Explaination%20+%20Results/1.%20PID%20Controller/Result1.jpg?raw=true)

### Position Graph
*Steering* is controlled by *Lateral Component* of PID Controller.
* Blue Line is the *expected position* of the vehicle.
* Orange Line is the *recorded position* of the vehicle.
#####
![Lateral PID Controller Graph](https://github.com/shivanshu1641/Drive.AI/blob/main/Explaination%20+%20Results/1.%20PID%20Controller/Result3.png?raw=true)

### Speed Graph
*Throttle* is controlled by *Longitudinal Component* of PID Controller.
* Blue Line is the *expected speed* at any given time.
* Orange Line is the *recored speed* at any given time.
#####
![Longitudinal PID Controller Graph](https://github.com/shivanshu1641/Drive.AI/blob/main/Explaination%20+%20Results/1.%20PID%20Controller/Result4.png?raw=true)
