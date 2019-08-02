# -*- coding: utf-8 -*-
"""
Created on Sat May 25 11:21:26 2019

@author: 熊熊熊宇达
"""

##########此代码用来转换yolo3 predict文件给tensorflow API调用COCO API计算mAP###########
import os
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from object_detection.metrics import coco_tools
import skimage.io as io
import pylab
import os,shutil

truth_path = '/Rudy/data/tensorflow-yolov3/mAP/ground-truth/'
pred_path = '/Rudy/data/tensorflow-yolov3/mAP/predicted/'
truth_save_path='/Rudy/data/tensorflow-yolov3/mAP/COCO_truth.json'
pred_save_path='/Rudy/data/tensorflow-yolov3/mAP/COCO_pred.json'

label_map_dict = [{'id': 0, 'name': 'ZW'},
                     {'id': 1, 'name': 'ST'},
                     {'id': 2, 'name': 'HD'},
                     {'id': 3, 'name': 'YJ'},
                     {'id': 4, 'name': 'DDW'},
                     {'id': 5, 'name': 'GS'},
                     {'id': 6, 'name': 'TZ'},
                     {'id': 7, 'name': 'JJ'}]

def convert_dt(path):
    label_map_dict = {'ZW': '0','ST':'1','HD':'2','YJ':'3','DDW':'4','GS':'5','TZ':'6','JJ':'7'}
    for _,_,names in os.walk(path):
        total_list = []
        print(names)
        for name in names:
            with open (path+name,'rb') as f:
                datas = f.readlines()
                data = [line.decode().strip().split() for line in datas]
                for i in range(len(data)):
                    data_list = []
                    data[i][4],data[i][5] = int(data[i][4]) - int(data[i][2]), int(data[i][5]) - int(data[i][3])
                    data_list.append(name.split('.')[0])
                    data_list= data_list + (data[i][2:]) + data[i][1:2]
                    data_list.append(label_map_dict[data[i][0]])
                    total_list.append(data_list)
    return np.array(total_list)

def convert_gt(path):
    label_map_dict = {'ZW': '0','ST':'1','HD':'2','YJ':'3','DDW':'4','GS':'5','TZ':'6','JJ':'7'}
    for _,_,names in os.walk(path):
        total_list = []
        print(names)
        for name in names:
            with open (path+name,'rb') as f:
                datas = f.readlines()
                data = [line.decode().strip().split() for line in datas]
                for i in range(len(data)):
                    data_list = []
                    data[i][3],data[i][4] = int(data[i][3]) - int(data[i][1]), int(data[i][4]) - int(data[i][2])
                    data_list.append(name.split('.')[0])
                    data_list = data_list + (data[i][1:])
                    data_list.append(1)
                    data_list.append(label_map_dict[data[i][0]])
                    total_list.append(data_list)  
    return np.array(total_list)

def convert_OBJ_DETE_dt(path):
    label_map_dict = {'ZW': '0','ST':'1','HD':'2','YJ':'3','DDW':'4','GS':'5','TZ':'6','JJ':'7'}
    for _,_,names in os.walk(path):
        image_ids = []
        detection_boxes = []
        detection_classes = []
        detection_scores = []

        for name in names:
            image_ids.append(name.split('.')[0])
            with open (path+name,'rb') as f:
                datas = f.readlines()
                data = [line.decode().strip().split() for line in datas]
                detection_box = []
                detection_class = []
                detection_score = []
                if len(data) == 0:
                    detection_box.append([0,0,0,0])
                    detection_class.append(0)
                    detection_score.append(0)

                for i in range(len(data)):

                    detection_box.append(data[i][2:])
                    detection_class.append(label_map_dict[data[i][0]])
                    detection_score.append(data[i][1])
            detection_boxes.append(np.array(detection_box, np.float))
            detection_classes.append(np.array(detection_class, np.int32))
            detection_scores.append(np.array(detection_score, np.float))
    return image_ids,detection_boxes,detection_scores,detection_classes
def convert_OBJ_DETE_gt(path):
    label_map_dict = {'ZW': '0','ST':'1','HD':'2','YJ':'3','DDW':'4','GS':'5','TZ':'6','JJ':'7'}
    for _,_,names in os.walk(path):

        image_ids1 = []
        ground_truth_boxes = []
        ground_truth_classes = []

        for name in names:
            image_ids1.append(name.split('.')[0])
            with open (path+name,'rb') as f:
                datas = f.readlines()
                data = [line.decode().strip().split() for line in datas]
                ground_truth_box = []
                ground_truth_class = []
                if len(data) == 0:
                    ground_truth_box.append([])
                for i in range(len(data)):

                    ground_truth_box.append(data[i][1:])
                    ground_truth_class.append(label_map_dict[data[i][0]])
            ground_truth_boxes.append(np.array(ground_truth_box, np.float))
            ground_truth_classes.append(np.array(ground_truth_class, np.int32))
    return image_ids1,ground_truth_boxes,ground_truth_classes

pred_image_ids,detection_boxes,detection_scores,detection_classes = convert_OBJ_DETE_dt(pred_path)
truth_image_ids,ground_truth_boxes,ground_truth_classes=convert_OBJ_DETE_gt(truth_path)
detections_list=coco_tools.ExportDetectionsToCOCO(pred_image_ids,detection_boxes,detection_scores,detection_classes,label_map_dict)
groundtruth_dict=coco_tools.ExportGroundtruthToCOCO(truth_image_ids,ground_truth_boxes,ground_truth_classes,label_map_dict)

groundtruth = coco_tools.COCOWrapper(groundtruth_dict)
detections = groundtruth.LoadAnnotations(detections_list)
evaluator = coco_tools.COCOEvalWrapper(groundtruth, detections)
metrics,ap = evaluator.ComputeMetrics()