import cv2
import os
import numpy as np

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

def config_yolov3(cfg):
    net = cv2.dnn.readNet(cfg.YOLOV3.WEIGHTS, cfg.YOLOV3.CFG)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return net, layer_names, output_layers

def config_detectron(cfg):
    path_weigth = cfg.DETECTRON.WEIGHTS
    path_config = cfg.DETECTRON.CFG
    confidences_threshold = cfg.DETECTRON.THRESHOLD
    num_of_class = cfg.DETECTRON.NUMBER_CLASS

    detectron = get_cfg()
    detectron.MODEL.DEVICE= cfg.DETECTRON.DEVICE
    detectron.merge_from_file(path_config)
    detectron.MODEL.WEIGHTS = path_weigth

    detectron.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidences_threshold
    detectron.MODEL.ROI_HEADS.NUM_CLASSES = num_of_class

    predictor = DefaultPredictor(detectron)

    return predictor