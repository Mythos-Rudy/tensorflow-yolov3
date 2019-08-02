# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 18:29:17 2019

@author: 熊熊熊宇达
"""

from __future__ import print_function

import requests
import numpy as np
from PIL import Image
import tensorflow as tf
from core import utils

# The server URL specifies the endpoint of your server running the ResNet
# model with the name "resnet" and using the predict interface.
SERVER_URL = 'http://192.168.1.99:8501/v1/models/cloth_yolov3'

# The image URL is the location of the image we should send to the server
IMAGE_URL = 'F:/servingtest/J01_2018.06.17 09_09_56.jpg'


def RESTmain():

  # Compose a JSON Predict request (send JPEG image in base64).
  image = Image.open(IMAGE_URL)
  image = np.array(image)
  predict_request = '{"IMAGE_INPUTS" : "%s"}' % image

  # Send few requests to warm-up the model.
  for _ in range(3):
    response = requests.post(SERVER_URL, data=predict_request)
    response.raise_for_status()

  # Send few actual requests and report average latency.
  total_time = 0
  num_requests = 10
  for _ in range(num_requests):
    response = requests.post(SERVER_URL, data=predict_request)
    response.raise_for_status()
    total_time += response.elapsed.total_seconds()
    prediction = response.json()

  print(prediction)
  print(total_time)
#  print('Prediction class: {}, avg latency: {} ms'.format(
#      prediction['classes'], (total_time*1000)/num_requests))




import grpc
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


#tf.app.flags.DEFINE_string('server', '192.168.1.99:8500',
#                           'PredictionService host:port')
#tf.app.flags.DEFINE_string('image', 'F:/servingtest/J01_2018.06.13 13_25_43.jpg', 'path to image in JPEG format')
#FLAGS = tf.app.flags.FLAGS


def gRPCmain():
  
  data = np.array(Image.open(IMAGE_URL), dtype=np.float32)

  channel = grpc.insecure_channel('192.168.1.99:8500')
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  # Send request
  # See prediction_service.proto for gRPC request/response details.
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'cloth_yolov3'
  request.model_spec.signature_name = 'serving_default'
  request.inputs['INPUT_IMAGES'].CopyFrom(
      tf.contrib.util.make_tensor_proto(data))
  result = stub.Predict(request, 10.0)  # 10 secs timeout
#  with tf.Session() as sess:
#      data = result.eval(session=sess)
  print(result.outputs['OUTPUT_BBOXES'].float_val)
  bboxes = np.array(result.outputs['OUTPUT_BBOXES'].float_val).reshape(-1,6)
  pred_img = utils.draw_bbox(data, bboxes)
  pred_img = Image.fromarray(np.uint8(pred_img))
  pred_img.save('d:/dog1.jpg')
  
#  pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
#                              np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
#                              np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)
#  bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
#  bboxes = utils.nms(bboxes, self.iou_threshold)


if __name__ == '__main__':
  gRPCmain()