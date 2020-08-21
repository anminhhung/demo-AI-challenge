import os
import cv2
import json
import random
import itertools
import numpy as np
import cv2

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg as config_detectron

from libs.detector.detectron.detector import predict, predictor

# cfg.merge_from_file('configs/detectron.yaml')

# path_weigth = cfg.DETECTRON.WEIGHTS
# path_weigth = "models/model_fasterRCNN_Khang_10k _46_v3.pth"
# path_config = "detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
# confidences_threshold = 0.6
# num_of_class = 5

# classes = ['Loai1', 'Loai2', 'Loai3', 'Loai4', 'Loai5']

# detectron = config_detectron()
# detectron.MODEL.DEVICE= 'cpu'
# detectron.merge_from_file(path_config)
# detectron.MODEL.WEIGHTS = path_weigth

# detectron.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidences_threshold
# detectron.MODEL.ROI_HEADS.NUM_CLASSES = num_of_class

# predictor = DefaultPredictor(detectron)

# def predict(image, predictor):
#     outputs = predictor(image)

#     boxes = outputs['instances'].pred_boxes
#     scores = outputs['instances'].scores
#     classes = outputs['instances'].pred_classes

#     list_boxes = []
#     list_scores = []
#     list_classes = []

#     for i in range(len(classes)):
#         if (scores[i] > 0.5):
#             for j in boxes[i]:
#                 x = int(j[0])
#                 y = int(j[1])
#                 w = int(j[2]) - x
#                 h = int(j[3]) - y

#             score = float(scores[i])
#             class_id = int(classes[i])
#             list_boxes.append([x, y, w, h])
#             list_scores.append(score)
#             list_classes.append(class_id)

#             cv2.rectangle(image, (x, y), (x+w, y+h), (random.randint(
#                 0, 255), random.randint(0, 255), 255), 1)

#     # return list_boxes, list_scores, list_classes
#     return image

if __name__ == '__main__':
    _frame = cv2.imread("frame.jpg")
    outputs = predict(_frame, predictor)

    cv2.imshow("frame", outputs)
    cv2.waitKey(0)