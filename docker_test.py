#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : evaluate.py
#   Author      : YunYang1994
#   Created date: 2019-02-21 15:30:26
#   Description :
#
#================================================================

import cv2
import os
import shutil
import numpy as np
import core.utils as utils
from core.config import cfg
from core.yolov3 import YOLOV3
import time

import json
import re
import shutil

import cv2
import os
import grpc
from matplotlib import pyplot as plt
#from django.views.decorators.http import require_http_methods
from grpc.framework.interfaces.face.face import AbortionError, ExpirationError
from grpc._channel import _Rendezvous

import io
from PIL import Image
import numpy as np
import time
import argparse
# import tensorflow as tf
from tensorflow.python.framework.tensor_util import make_tensor_proto
# from object_detection.utils import label_map_util
# from tensorflow_serving_client.proto_util import copy_message
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc


channel = grpc.insecure_channel("192.168.1.33:9002")
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
request = predict_pb2.PredictRequest()
# 这里由保存和运行时定义，第一个是运行时配置的模型名，第二个是保存时输入的方法名
request.model_spec.name = 'faster_50'
# 入参参照入参定义
request.model_spec.signature_name = "detection_signature"

class YoloTest(object):
    def __init__(self):
        self.input_size       = cfg.TEST.INPUT_SIZE
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes          = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes      = len(self.classes)
        self.anchors          = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.score_threshold  = cfg.TEST.SCORE_THRESHOLD
        self.iou_threshold    = cfg.TEST.IOU_THRESHOLD
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.annotation_path  = cfg.TEST.ANNOT_PATH
        self.weight_file      = cfg.TEST.WEIGHT_FILE
        self.write_image      = cfg.TEST.WRITE_IMAGE
        self.write_image_path = cfg.TEST.WRITE_IMAGE_PATH
        self.show_label       = cfg.TEST.SHOW_LABEL

#        with tf.name_scope('input'):
#            self.input_data = tf.placeholder(dtype=tf.float32, name='input_data')
#            self.trainable  = tf.placeholder(dtype=tf.bool,    name='trainable')
#
#        model = YOLOV3(self.input_data, self.trainable)
#        self.pred_sbbox, self.pred_mbbox, self.pred_lbbox = model.pred_sbbox, model.pred_mbbox, model.pred_lbbox
#
#        with tf.name_scope('ema'):
#            ema_obj = tf.train.ExponentialMovingAverage(self.moving_ave_decay)
#
#        self.sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
#        self.saver = tf.train.Saver(ema_obj.variables_to_restore())
#        #self.saver.restore(self.sess, self.weight_file)
#
#    def predict(self, image):
#
#        org_image = np.copy(image)
#        org_h, org_w, _ = org_image.shape
#
#        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
#        image_data = image_data[np.newaxis, ...]
#        start_time = time.time()
#
#        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run(
#            [self.pred_sbbox, self.pred_mbbox, self.pred_lbbox],
#            feed_dict={
#                self.input_data: image_data,
#                self.trainable: False
#            }
#        )
#        time_cost = time.time() - start_time
#
#        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
#                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
#                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)
#        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
#        bboxes = utils.nms(bboxes, self.iou_threshold)
#
#        return bboxes,time_cost

    def evaluate(self):
        predicted_dir_path = './mAP1/predicted'
        ground_truth_dir_path = './mAP1/ground-truth'
        save_img_path = './mAP1/images/'
        if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
        if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
        if os.path.exists(self.write_image_path): shutil.rmtree(self.write_image_path)
        os.mkdir(predicted_dir_path)
        os.mkdir(ground_truth_dir_path)
        os.mkdir(self.write_image_path)

        with open(self.annotation_path, 'r') as annotation_file:
            run_time_cost = 0
            for num, line in enumerate(annotation_file):
                annotation = line.strip().split()
                image_path = annotation[0]
                image_name = image_path.split('/')[-1]
                image = cv2.imread(image_path)
                bbox_data_gt = np.array([list(map(float, box.split(','))) for box in annotation[1:]]).astype(np.int)

                if len(bbox_data_gt) == 0:
                    bboxes_gt=[]
                    classes_gt=[]
                else:
                    bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
                ground_truth_path = os.path.join(ground_truth_dir_path, str(num)+'-'+image_name + '.txt')

                print('=> ground truth of %s:' % image_name)
                num_bbox_gt = len(bboxes_gt)
                classes_dict = ['WR','ST','HD','ZH','PD','GS','ZK']
                with open(ground_truth_path, 'w') as f:
                    for i in range(num_bbox_gt):
#                        class_name = self.classes[classes_gt[i]]
                        class_name = classes_dict[classes_gt[i]]
                        xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                        bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                        f.write(bbox_mess)
                        print('\t' + str(bbox_mess).strip())
                print('=> predict result of %s:' % image_name)
                predict_result_path = os.path.join(predicted_dir_path, str(num)+'-'+image_name + '.txt')
                cv2.imwrite(save_img_path+str(num)+'-'+image_name, image)
                
                image_np = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                request.inputs['inputs'].CopyFrom(make_tensor_proto(image_np_expanded))
                result = stub.Predict(request, 40.0)
                boxes = result.outputs['detection_boxes'].float_val
                scores = result.outputs['detection_scores'].float_val
                classes = result.outputs['detection_classes'].float_val
                boxes = np.asarray(boxes).reshape(-1, 4)
                scores = np.asarray(scores)
                classes = np.asarray(classes)-1
                boxes = np.column_stack((boxes,scores,classes))
                bboxes_pr = boxes
                if self.write_image:
                    image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
                    cv2.imwrite(self.write_image_path+image_name, image)

                with open(predict_result_path, 'w') as f:
                    for bbox in bboxes_pr:
                        if bbox[4] >= 0.5:
                            bbox[0],bbox[2] = bbox[0]*800,bbox[2]*800
                            bbox[1],bbox[3] = bbox[1]*1280,bbox[3]*1280
                            coor = np.array(bbox[:4], dtype=np.int32)
                            score = bbox[4]
                            class_ind = int(bbox[5])
                            class_name = classes_dict[class_ind]
                            print(class_ind,class_name)
                            score = '%.4f' % score
                            ymin, xmin, ymax, xmax = list(map(str, coor))
                            bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                            f.write(bbox_mess)
                            print('\t' + str(bbox_mess).strip())
        return run_time_cost


if __name__ == '__main__': 
    run_time_cost = YoloTest().evaluate()
    print('total: ',run_time_cost,'s')
    print('fps: ',run_time_cost/179)




