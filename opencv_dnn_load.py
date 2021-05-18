# -*- coding: utf-8 -*-
import torch
import cv2
import numpy as np
# 1, 转.换pt模型至.weights
# $ python3  -c "from models import *; convert('cfg/yolov3-spp-terahertz-8cls.cfg', 'weights/best_terahertz-yolov3-spp.pt')"

# 2 load darknet model
cfg = 'cfg/yolov3-spp-terahertz-8cls.cfg'
weight = 'weights/best_terahertz-yolov3-spp.weights'
# cfg = 'cfg/yolov3.cfg'
# weight = 'weights/yolov3.weights'

thr = 0.5
nms = 0.6

# 3. test model
# img_file = '/media/disk4T/DeepLearningProjects/Terahertz/dataset/images/1.png' # screw drive
# class_file = '/media/disk4T/DeepLearningProjects/Terahertz/dataset/classes.txt'

img_file = '/media/disk4T/DeepLearningProjects/Terahertz/dataset/images/1.png'
# class_file = '/media/gaoya/disk/DeepLearning/darknet/data/coco.names'
class_file = r"/media/disk4T/DeepLearningProjects/Terahertz/dataset/classes.txt"
with open(class_file,'rt') as f:
    names = f.read().rstrip('\n').split('\n')

# 加载yolo模型
net = cv2.dnn_DetectionModel(weight, cfg)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA) 
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
net.setInputSize(512,512) # 设置网络输入尺寸
net.setInputScale(1.0/255)
net.setInputSwapRB(True)

frame=cv2.imread(img_file)

import time
s = time.time()
for _ in range(10):
    classes, confs, boxes = net.detect(frame, 0.5, 0.5)
print('平均推理时间:', (time.time() - s) / 10.0)
classes, confs, boxes = net.detect(frame, 0.5, 0.5)

for id, conf, box in zip(classes.flatten(), confs.flatten(), boxes):
    label = '{}, {:.2f}'.format(names[id], conf)
    # print(label)
    labelsize, baseLine= cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
    left, top, width, height = box
    top = max(top, labelsize[1])
    cv2.rectangle(frame, box, color=(0, 255, 0), thickness=3)
    cv2.rectangle(frame, (left, top-labelsize[1]),
                 (left+labelsize[0], top+baseLine),(255, 255, 255), cv2.FILLED)
    cv2.putText (frame, label,(left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0, 0))

cv2.imshow('frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()