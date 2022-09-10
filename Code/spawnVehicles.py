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
import random

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
import time
import math
import numpy as np
import cv2

actor_list = []
# model = torch.hub.load('.\yolov5\yolov5', 'custom', path='cars-1.pt', source='local')  
# print("Loaded YOLOv5 Model")
spawn_id = 24
def detect(model, img):
    imgs = [img]  
    model.conf = 0.6
    results = model(imgs, size=416)
    xmin = list(results.pandas().xyxy[0]['xmin'])
    xmax = list(results.pandas().xyxy[0]['xmax'])
    ymin = list(results.pandas().xyxy[0]['ymin'])
    ymax = list(results.pandas().xyxy[0]['ymax'])
    conf = list(results.pandas().xyxy[0]['confidence'])
    print(results.pandas().xyxy[0])
    c = list(results.pandas().xyxy[0]['name'])
    print(c)
    x = imgs[0].shape[0]
    y = imgs[0].shape[1]
    for i in range(0,len(xmin)):
        cv2.rectangle(imgs[0], (int(xmin[i]), int(ymin[i])), (int(xmax[i]), int(ymax[i])), (0, 0, 220), 8)
        cv2.putText(imgs[0], c[i]+" "+str(int(conf[i]*100))+"%", (int(xmin[i]), int(ymin[i])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 220), 2)
        #print(classes[int(c[i])], str(int(conf[i]*100))+"%")
    print(results.imgs[0].shape)        
    out = cv2.resize(results.imgs[0],(y,x),interpolation = cv2.INTER_AREA)[..., ::-1]
    return out

def show_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((416, 416, 4))
    i3 = i2[:, :, :3]
    frame = detect(model, i3)
    cv2.imshow("", frame)
    cv2.waitKey(1)

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []


try:
    time.sleep(2)
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(2.0)
    print(client.get_available_maps())
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    print(world)
    al = world.get_actors()
    print(al)
    vehicle = None
    for i in al:
        if ("role_name" in i.attributes):
            if (i.attributes["role_name"] == "hero"):
                vehicle = i
    bp = blueprint_library.filter('cybertruck')[0]
    #bp.set_attribute('role_name', 'hero')
    sim_world = client.get_world()
    waypoints = sim_world.get_map().generate_waypoints(2.0)
    spawn_points = world.get_map().get_spawn_points()
    vehicle = world.spawn_actor(bp, spawn_points[spawn_id])
    # for w in waypoints:
    #         traffic_lights = sim_world.get_traffic_lights_from_waypoint(w, 5.0)
    #         for traffic_light in traffic_lights:
    #             traffic_light.set_state(carla.TrafficLightState.Green)
    #             traffic_light.set_green_time(12000)
    actor_list.append(vehicle)
    vehicle.set_autopilot(True)
    time.sleep(4)
    source = spawn_points[spawn_id]
    number_of_spawn_points = len(spawn_points)
    #random.seed(2)
    random.seed(3)
    random.shuffle(spawn_points)
    blueprints = get_actor_blueprints(world, 'vehicle.*', 'All')
    blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
    # for i in range(0,50):
    #     print(i)
    #     if(spawn_points[i] == source ):
    #         continue
    #     vehicle = world.spawn_actor(blueprints[i%len(blueprints)], spawn_points[i])
    #     actor_list.append(vehicle)
    #     vehicle.set_autopilot(True)


    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    SetVehicleLightState = carla.command.SetVehicleLightState
    FutureActor = carla.command.FutureActor
    batch = []
    for i in range(0,50):
        if(spawn_points[i] == source):
            continue
        batch.append(SpawnActor(blueprints[i%len(blueprints)], spawn_points[i])
                .then(SetAutopilot(FutureActor, True)))
    for response in client.apply_batch_sync(batch, False):
            if response.error:
                print("Error")
            else:
                actor_list.append(response.actor_id)

    time.sleep(3)
    for w in waypoints:
            traffic_lights = sim_world.get_traffic_lights_from_waypoint(w, 5.0)
            for traffic_light in traffic_lights:
                traffic_light.set_state(carla.TrafficLightState.Red)
    # blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
    # blueprint.set_attribute('image_size_x', '416')
    # blueprint.set_attribute('image_size_y', '416')
    # blueprint.set_attribute('fov', '110')
    # blueprint.set_attribute('sensor_tick', '1.0')
    # transform = carla.Transform(carla.Location(x=0.8, z=1.7))
    # sensor = world.spawn_actor(blueprint, transform, attach_to=vehicle)
    # sensor.listen(lambda data: show_img(data))
    # time.sleep(90)
    # # print(vehicle.__setattr__('isTraffic', "role") )
    # print(vehicle.__getattribute__('isTraffic' ))
    # #vehicle.set_attribute('role', '1234') '__setattr__', 
    #print(vehicle.attributes)
    print(actor_list)
    time.sleep(400)
finally:
    print(actor_list)
    print('destroying actors')
    print(len(actor_list))
    for i in range(0,len(actor_list)):
        print("destroyed")
        actor_list[i].destroy()
        time.sleep(1)
    print('done.')
    #time.sleep(10)