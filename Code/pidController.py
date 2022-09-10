import sys
import glob
import os
import pandas as pd

import torch
import cv2
from PIL import Image
from flask import Flask, request
from flask_restful import reqparse
from subprocess import check_output, STDOUT
import numpy as np
import skimage.exposure as exposure


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

sys.path.append("C://Carla//CARLA_0.9.12//WindowsNoEditor//PythonAPI//carla//agents//navigation")
sys.path.append("C://Carla//CARLA_0.9.12//WindowsNoEditor//PythonAPI//carla")

import carla
import controller2
import time
import math
print("Loaded Modules")

model = torch.hub.load('.\yolov5\yolov5', 'custom', path='CarlaObject-2.pt', source='local')  
print("Loaded YOLOv5 Model")
actor_list = []
count = 0

spawn_no = 24
iteration = 13
trajectory = 'trajectory.csv'

def get_speed(vehicle):
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


try:
    print(dir(controller2))
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(2.0)
    print(client.get_available_maps())
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    bp = blueprint_library.filter('model3')[0]
    bp.set_attribute('role_name', 'hero')
    spawn_point = world.get_map().get_spawn_points()[spawn_no]
    vehicle = world.spawn_actor(bp, spawn_point)
    actor_list.append(vehicle)
    blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
    blueprint.set_attribute('image_size_x', '416')
    blueprint.set_attribute('image_size_y', '416')
    blueprint.set_attribute('fov', '110')
    blueprint.set_attribute('sensor_tick', '0.28')
    transform = carla.Transform(carla.Location(x=0, z=2.4))
    #time.sleep(40)
    pid = controller2.VehiclePIDController(vehicle, {"K_P": 1, "K_D": 0.2, "K_I": 0.01},{"K_P": 1, "K_D": 0.2, "K_I": 0.01})
    df = pd.read_csv(trajectory, names = ["x","y","speed"])
    x = list(df["x"])
    y = list(df["y"])
    speed = list(df["speed"])
    waypoints = []
    for i in range(0,len(x)):
        waypoints.append({"x": x[i], "y": y[i]})
        
    print(df)
    for waypoint in waypoints:
        loc = carla.Location(x = waypoint["x"], y = waypoint["y"])
        world.debug.draw_string(loc, 'O', draw_shadow=False,
                                        color=carla.Color(r=0, g=255, b=0), life_time=180.0,
                                        persistent_lines=True)
    d = 2
    time.sleep(4)
    i = 110
    final = carla.Location(x = waypoints[-1]["x"], y = waypoints[-1]["y"])
    data = []
    data2 = []
    data3 = []
    while(i < len(waypoints)-1):
        veh_loc = vehicle.get_location()
        data.append((veh_loc.x,veh_loc.y,get_speed(vehicle)))
        data2.append((waypoints[i]["x"], waypoints[i]["y"], speed[i]))
        if(final.distance(vehicle.get_location())<4):
            break
        loc = carla.Location(x = waypoints[i]["x"], y = waypoints[i]["y"])
        if((loc.distance(vehicle.get_location())) < 4 or (loc.distance(vehicle.get_location())) > 6) and i>50:
            i = i-2
            if(i >= len(waypoints)-5):
                break
            continue
        if(i >= len(waypoints)-5):
                break
        data3.append((speed[i-4], get_speed(vehicle)))
        control = pid.run_step(speed[i], waypoints[i])
        vehicle.apply_control(control)
        world.debug.draw_string(loc, 'O', draw_shadow=False,
                                        color=carla.Color(r=255, g=0, b=0), life_time=5.0,
                                        persistent_lines=True)
        i = i+4
        time.sleep(0.035)
    print("Done")
finally:
    df = pd.DataFrame(data=data, columns=["x","y","speed"])
    df.to_csv("vehicle_trajectory-"+str(iteration)+".csv")
    df = pd.DataFrame(data=data2, columns=["x","y","speed"])
    df.to_csv("reference_trajectory-"+str(iteration)+".csv")
    df = pd.DataFrame(data=data3, columns=["speed1","speed2"])
    df.to_csv("reference_speed-"+str(iteration)+".csv")
    
    for actor in actor_list:
        actor.destroy()
    print('done.')