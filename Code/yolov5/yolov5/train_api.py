import torch
import cv2
from PIL import Image
from flask import Flask, request
from flask_restful import reqparse
from subprocess import check_output, STDOUT
import train
import os
import pandas as pd

def getEvalScore():
  count = len(getDir("./runs/train/"))
  results_path = "./runs/train/exp/results.csv"
  if(count>1):
    results_path = "./runs/train/exp"+str(count)+"/results.csv"
  results = pd.read_csv(results_path)
  results = results.to_dict('list')
  recall = results["      metrics/recall"][-1]
  prec = results["   metrics/precision"][-1]
  map05 = results["     metrics/mAP_0.5"][-1]
  map095 = results["metrics/mAP_0.5:0.95"][-1]
  box_loss = results["      train/box_loss"][-1]
  obj_loss = results["      train/obj_loss"][-1]
  cls_loss = results["      train/cls_loss"][-1]
  total_loss = box_loss + obj_loss + cls_loss
  if(total_loss == 0):
    total_loss = -1
  total_map = (0.8*map05)+(0.2*map095)
  eval_score = (0.3*prec)+(0.3*recall)+(0.3*total_map)+(0.001*(1/total_loss))
  return eval_score





def getDir(path):
  return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

def trainModel(dataset, source_loc = "",dest_loc = "",epochs=100, imgsize=416, from_scratch = True):
  bucketName = source_loc
  datasetName = dataset
  dwnImg = "gsutil cp -r gs://"+bucketName+"/"+datasetName + " " + "./"
  check_output(dwnImg, stderr=STDOUT, shell=True)
  dataset_path = "./"+datasetName+"/data.yaml"  
  print(dataset_path)    
  print(from_scratch)
  eval_score = 0
  hyp_path = "./data/hyps/hyp.scratch.yaml"
  threshold = 0.6
  weights_path = "ip102-combined.pt"
  if(from_scratch is True):
    weights_path = "yolov5s.pt"
  #train.run(data=dataset_path, imgsz=imgsize, weights=weights_path, epochs = 20)
  for i in range(0,epochs, 10):
    train.run(data=dataset_path, imgsz=imgsize, weights=weights_path, epochs = 10, patience = 5, hyp = hyp_path)
    new_eval_score = getEvalScore()
    print(new_eval_score)
    if(new_eval_score < 0.3*threshold ):
      hyp_path = "./data/hyps/hyp.scratch-med.yaml"
    elif(new_eval_score > 0.3*threshold and new_eval_score<0.5*threshold):
      hyp_path = "./data/hyps/hyp.scratch-low.yaml"
    elif(new_eval_score > 0.5*threshold and new_eval_score< 0.7*threshold):
      hyp_path = "./data/hyps/hyp.scratch.yaml"
    elif(new_eval_score >= 0.7*threshold):
      print(new_eval_score)
      print("Achieved satisfactory result!")
      break
    count = len(getDir("./runs/train/"))
    weights_path = "./runs/train/exp/weights/best.pt"
    if(count>1):
      weights_path = "./runs/train/exp"+str(count)+"/weights/best.pt"  
  count = len(getDir("./runs/train/"))
  weights_path = "./runs/train/exp/weights/best.pt"
  results_path = "./runs/train/exp/results.csv"
  if(count>1):
    weights_path = "./runs/train/exp"+str(count)+"/weights/best.pt"
    results_path = "./runs/train/exp"+str(count)+"/results.csv"
  print(weights_path)
  new_weights_path = "/".join(weights_path.split("/")[0:-1])+"/"+datasetName+"-best.pt"
  print(new_weights_path)
  os.rename(weights_path,new_weights_path) 
  uploadImg = "gsutil cp "+new_weights_path+" gs://"+bucketName
  check_output(uploadImg, stderr=STDOUT, shell=True)
  results = pd.read_csv(results_path)
  return results.to_dict('list')


app_flask = Flask(__name__)
parser = reqparse.RequestParser()
parser.add_argument('datasetName', type=str, required=True, help="Enter dataset name")
parser.add_argument('epochs', type=int, required=True, help="Enter number of epochs")
parser.add_argument('source_loc', type=str, required=True, help="Enter source")
parser.add_argument('dest_loc', type=str, required=True, help="Enter destination")
parser.add_argument('from_scratch', type=int, required=True, help="Enter from scratch value")


@app_flask.route('/train', methods=['GET', 'POST'])
def train_model():
    if request.method == 'GET' or request.method == 'POST':        
        args = parser.parse_args()
        datasetName = args['datasetName']
        epochs = args["epochs"]
        source_loc = args["source_loc"]
        dest_loc = args["dest_loc"]
        from_scratch = args["from_scratch"]
        if(from_scratch == 1):
          from_scratch = True
        else:
          from_scratch = False
        imgsize = 416
        results = trainModel(datasetName, source_loc,dest_loc,epochs,imgsize,from_scratch)
        return results

if __name__ == '__main__':
    app_flask.run(debug=True)
    #print(getEvalScore())