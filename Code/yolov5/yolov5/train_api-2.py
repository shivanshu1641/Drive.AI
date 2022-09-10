import torch
import cv2
from PIL import Image
from flask import Flask, request
from flask_restful import reqparse
from subprocess import check_output, STDOUT
import train
import os
import pandas as pd



def getDir(path):
  return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

def trainModel(dataset, source_loc = "",dest_loc = "",epochs=1, imgsize=416, from_scratch = True):
  #print(locations[dataset])
  bucketName = source_loc
  datasetName = dataset
  dwnImg = "gsutil cp -r gs://"+bucketName+"/"+datasetName + " " + "./"
  #check_output(dwnImg, stderr=STDOUT, shell=True)
  dataset_path = "./"+datasetName+"/data.yaml"  
  print(dataset_path)    
  if(from_scratch == "True"):
    train.run(data=dataset_path, imgsz=imgsize, weights='./yolov5/yolov5/yolov5s.pt', epochs = epochs)
  else:
    train.run(data=dataset_path, imgsz=imgsize, weights='./yolov5/yolov5/ip102-combined.pt', epochs = epochs)
  count = len(getDir("./yolov5/yolov5/runs/train"))
  weights_path = "./yolov5/yolov5/runs/train/exp/weights/best.pt"
  results_path = "./yolov5/yolov5/runs/train/exp/results.csv"
  if(count>1):
    weights_path = "./yolov5/yolov5/runs/train/exp"+str(count)+"/weights/best.pt"
    results_path = "./yolov5/yolov5/runs/train/exp"+str(count)+"/results.csv"
  print(weights_path)
  uploadImg = "gsutil cp "+weights_path+" gs://"+bucketName
  #check_output(uploadImg, stderr=STDOUT, shell=True)
  results = pd.read_csv(results_path)
  return results.to_dict('list')


app_flask = Flask(__name__)
parser = reqparse.RequestParser()
parser.add_argument('datasetName', type=str, required=True, help="Enter dataset name")
parser.add_argument('epochs', type=int, required=True, help="Enter number of epochs")
parser.add_argument('source_loc', type=str, required=True, help="Enter source")
parser.add_argument('dest_loc', type=str, required=True, help="Enter destination")
parser.add_argument('from_scratch', type=bool, required=True, help="Enter from scratch value")

@app_flask.route('/',methods=['GET','POST'])
def print_hello():
  return "Hello"
@app_flask.route('/train', methods=['GET', 'POST'])
def train_model():
    if request.method == 'GET' or request.method == 'POST':        
        print("Yes")
        args = parser.parse_args()
        datasetName = args['datasetName']
        epochs = args["epochs"]
        source_loc = args["source_loc"]
        dest_loc = args["dest_loc"]
        from_scratch = args["from_scratch"]
        imgsize = 416
        results = trainModel(datasetName, source_loc,dest_loc,epochs,imgsize,from_scratch)
        return results

if __name__ == '__main__':
    app_flask.run(host='0.0.0.0', debug=True)