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
print("Hello")


model = torch.hub.load('.\yolov5\yolov5', 'custom', path='carlaObject-2.pt', source='local')  
print("Loaded YOLOv5 Model")


spawn_id = 24
trajectory_id = 9


applyBrake = False
TARGET_SPEED_FORWARD = 15
TARGET_SPEED_TURN = 12
TARGET_SPEED_SLOW = 5
OBSTACLE_AHEAD = False

def get_target_speed(l, w):
    global applyBrake
    v_x = l.x
    v_y = l.y
    w_x = w["x"]
    w_y = w["y"]
    diff_x = abs(v_x - w_x)
    diff_y = abs(v_y - w_y)
    if(not applyBrake):
        if(OBSTACLE_AHEAD):
            print("SLowing Down")
            return TARGET_SPEED_SLOW
    if(diff_x < 0.5 or diff_y < 0.5):
        #print("Forward")
        return TARGET_SPEED_FORWARD
    else:
        #print("Turn")
        return TARGET_SPEED_TURN


def getTrafficLightColor(img):
    histGR = cv2.calcHist([img], [1, 2], None, [256, 256], [0, 256, 0, 256])

    # histogram is float and counts need to be scale to range 0 to 255
    histScaled = exposure.rescale_intensity(histGR, in_range=(0,1), out_range=(0,255)).clip(0,255).astype(np.uint8)

    # make masks
    ww = 256
    hh = 256
    ww13 = ww // 3
    ww23 = 2 * ww13
    hh13 = hh // 3
    hh23 = 2 * hh13
    black = np.zeros_like(histScaled, dtype=np.uint8)
    # specify points in OpenCV x,y format
    ptsUR = np.array( [[[ww13,0],[ww-1,hh23],[ww-1,0]]], dtype=np.int32 )
    redMask = black.copy()
    cv2.fillPoly(redMask, ptsUR, (255,255,255))
    ptsBL = np.array( [[[0,hh13],[ww23,hh-1],[0,hh-1]]], dtype=np.int32 )
    greenMask = black.copy()
    cv2.fillPoly(greenMask, ptsBL, (255,255,255))

    #Test histogram against masks
    region = cv2.bitwise_and(histScaled,histScaled,mask=redMask)
    redCount = np.count_nonzero(region)
    region = cv2.bitwise_and(histScaled,histScaled,mask=greenMask)
    greenCount = np.count_nonzero(region)
    
    # Find color
    threshCount = 1
    if redCount >= greenCount and redCount >= threshCount:
        color = "red"
    elif greenCount >= redCount and greenCount >= threshCount:
        color = "green"
    elif redCount < threshCount and greenCount < threshCount:
        color = "yellow"
    else:
        color = "other"
    print(color)
    return color


actor_list = []
count = 0



def get_speed(vehicle):
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


def area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1)
                + x3 * (y1 - y2)) / 2.0)

def isInside(x1, y1, x2, y2, x3, y3, x, y):
    A = area (x1, y1, x2, y2, x3, y3)
    A1 = area (x, y, x2, y2, x3, y3)
    A2 = area (x1, y1, x, y, x3, y3)
    A3 = area (x1, y1, x2, y2, x, y)
    if(A == A1 + A2 + A3):
        return True
    else:
        return False

def inLane(xmin,ymin,xmax,ymax,x1,x2,y1,y2,x3,y3):
    if(isInside(x1, y1, x2, y2, x3, y3, xmin, ymin) and isInside(x1, y1, x2, y2, x3, y3, xmax, ymax)):
        return True
    else:
        return False


def detect(model, img):
    imgs = [img]  
    model.conf = 0.5
    results = model(imgs, size=416)
    isTraffic = -1
    xmin = list(results.pandas().xyxy[0]['xmin'])
    xmax = list(results.pandas().xyxy[0]['xmax'])
    ymin = list(results.pandas().xyxy[0]['ymin'])
    ymax = list(results.pandas().xyxy[0]['ymax'])
    conf = list(results.pandas().xyxy[0]['confidence'])
    c = list(results.pandas().xyxy[0]['name'])
    #print(c)
    x = imgs[0].shape[0]
    y = imgs[0].shape[1]
    global count
    global applyBrake
    global OBSTACLE_AHEAD
    if(len(xmin)>0):
        print("Traffic-Light")
        print(results.pandas().xyxy[0])
        m = 0
        area = 0
        for i in range(0,len(xmin)):
            if(ymax[i]>y):
                ymax[i] = y
            if(xmin[i]>x):
                xmin[i] = x
            a = (ymax[i] - ymin[i])*(xmax[i] - xmin[i])
            print(c[i])
            print(a)
            x1 = x//2+20
            y1 = y//3-30
            x2 = x//4-90
            y2 = y
            x3 = x+10
            y3 = y
            sameLane = inLane(xmin[i],ymin[i],xmax[i],ymax[i],x1,x2,y1,y2,x3,y3)
            if(a>1000 and a<1200 and c[i]=="vehicle" and sameLane):
                print("In same Lane")
                OBSTACLE_AHEAD = True
            elif(a>1200 and c[i] == "vehicle" and sameLane):
                print("In Same Lane")
                print("Applying Brake..")
                applyBrake = True
                break
            else:
                OBSTACLE_AHEAD = False
            if(c[i] != "traffic_light"):
                continue
            print("yes")
            if(a>area):
                m = i
        if(a!=0 and c[m] == "traffic_light"):
            traffic_light = img[int(ymin[m]):int(ymax[m]), int(xmin[m]):int(xmax[m])]
        #traffic_light = cv2.cvtColor(traffic_light)   
        #cv2.imwrite('t-'+str(count)+'.jpg',img)
            if(getTrafficLightColor(traffic_light) == 'red'):
                applyBrake = True
            else:
                applyBrake = False
    else:
        OBSTACLE_AHEAD = False
        applyBrake = False
    count = count + 1
    # for i in range(0,len(xmin)):
    #     cv2.rectangle(imgs[0], (int(xmin[i]), int(ymin[i])), (int(xmax[i]), int(ymax[i])), (0, 0, 220), 8)
    #     cv2.putText(imgs[0], c[i]+" "+str(int(conf[i]*100))+"%", (int(xmin[i]), int(ymin[i])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 220), 2)
    #     #print(classes[int(c[i])], str(int(conf[i]*100))+"%")
    # print(results.imgs[0].shape)        
    # out = cv2.resize(results.imgs[0],(y,x),interpolation = cv2.INTER_AREA)[..., ::-1]
    # return out




def show_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((416, 416, 4))
    i3 = i2[:, :, :3]
    detect(model, i3)
    #cv2.imshow("", frame)
    #cv2.waitKey(1)


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
    print(dir(controller2))
    client = carla.Client('127.0.0.1', 2000)
    port = 8000
    seed_value = 0
    tm = client.get_trafficmanager(port)
    tm.set_random_device_seed(seed_value)
    client.set_timeout(2.0)
    print(client.get_available_maps())
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    bp = blueprint_library.filter('model3')[0]
    bp.set_attribute('role_name', 'hero')
    spawn_point = world.get_map().get_spawn_points()[spawn_id]
    vehicle = world.spawn_actor(bp, spawn_point)
    print(" --------------------------------------------- Spawned Vehicle  ---------------------------------------------")
    actor_list.append(vehicle)
    points = world.get_map().get_spawn_points()
    print(points[0].transform)
    for i in range(0,len(points)):
        loc = points[i].location
        world.debug.draw_string(loc, str(i), draw_shadow=False,
                                         color=carla.Color(r=255, g=0, b=0), life_time=600.0,
                                         persistent_lines=True)

    blueprint_library = world.get_blueprint_library()
    blueprints = get_actor_blueprints(world, 'vehicle.*', 'All')
    banned_vehicles = ['ambulance', 'firetruck', 'carlacola','microlino']
    blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
    blueprints = [x for x in blueprints if x.tags[-1] not in banned_vehicles]
    vehicle_spawn_points = [100,45,56,3,65,66,73,0,29,78,93,33,91,46,47]
    for b in blueprints:
        print(b.tags)
    
    blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
    blueprint.set_attribute('image_size_x', '416')
    blueprint.set_attribute('image_size_y', '416')
    blueprint.set_attribute('fov', '110')
    blueprint.set_attribute('sensor_tick', '0.28')
    transform = carla.Transform(carla.Location(x=0, z=2.4))
    sensor = world.spawn_actor(blueprint, transform, attach_to=vehicle)
    sensor.listen(lambda data: show_img(data))
    
    #time.sleep(40)
    pid = controller2.VehiclePIDController(vehicle, {"K_P": 1, "K_D": 0.2, "K_I": 0.01},{"K_P": 1, "K_D": 0.2, "K_I": 0.01})
    df = pd.read_csv('trajectory.csv', names = ["x","y","speed"])
    #df = pd.read_csv('trajectory-'+str(trajectory_id)+'.csv', names = ["x","y","speed"])
    x = list(df["x"])
    y = list(df["y"])
    speed = list(df["speed"])
    waypoints = []
    for i in range(0,len(x)):
        waypoints.append({"x": x[i], "y": y[i]})
        
    print(df)
    print(waypoints)
    print(speed)
    print(pid.run_step(speed[100], waypoints[100]))
    for waypoint in waypoints:
         loc = carla.Location(x = waypoint["x"], y = waypoint["y"])
         world.debug.draw_string(loc, 'O', draw_shadow=False,
                                         color=carla.Color(r=0, g=255, b=0), life_time=180.0,
                                         persistent_lines=True)
    d = 2
    time.sleep(9)
    i = 200
    #i = 200
    final = carla.Location(x = waypoints[-1]["x"], y = waypoints[-1]["y"])
    data = []
    data2 = []
    data3 = []

    tm_port = tm.get_port()

    for v in range(0,len(vehicle_spawn_points)):
        spawn_point = world.get_map().get_spawn_points()[vehicle_spawn_points[v]]
        print(spawn_point)
        vehicle1 = world.spawn_actor(blueprints[v%len(blueprints)], spawn_point)
        vehicle1.set_autopilot(True, tm_port)
        actor_list.append(vehicle1)





    while(i < len(waypoints)-1):
        if(applyBrake):
            vehicle.apply_control(carla.VehicleControl(brake=1.0))
            time.sleep(2)
            #applyBrake = False
            continue
        veh_loc = vehicle.get_location()
        data.append((veh_loc.x,veh_loc.y,get_speed(vehicle)))
        data2.append((waypoints[i]["x"], waypoints[i]["y"], speed[i]))
        if(final.distance(vehicle.get_location())<4):
            break
        loc = carla.Location(x = waypoints[i]["x"], y = waypoints[i]["y"])
        if((loc.distance(vehicle.get_location())) < 4 or (loc.distance(vehicle.get_location())) > 6) and i>120:
            i = i-2
            if(i >= len(waypoints)-5):
                break
            continue
        if(i >= len(waypoints)-5):
                break
        next_speed = get_target_speed(veh_loc, waypoints[i])
        data3.append((speed[i-4], get_speed(vehicle)))
        control = pid.run_step(next_speed, waypoints[i])
        #print(i, speed[i])
        vehicle.apply_control(control)
        world.debug.draw_string(loc, 'O', draw_shadow=False,
                                        color=carla.Color(r=255, g=0, b=0), life_time=5.0,
                                        persistent_lines=True)
        i = i+4
        time.sleep(0.035)
    print("Done")
finally:
    #print(vehicle.__getattribute__('isTraffic' ))
    df = pd.DataFrame(data=data, columns=["x","y","speed"])
    df.to_csv("vehicle_trajectory-9.csv")
    df = pd.DataFrame(data=data2, columns=["x","y","speed"])
    df.to_csv("reference_trajectory-9.csv")
    df = pd.DataFrame(data=data3, columns=["speed1","speed2"])
    df.to_csv("reference_speed-9.csv")
    
    for actor in actor_list:
        actor.destroy()
    print('done.')