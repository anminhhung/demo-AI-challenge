from .config_detector import config_yolov3, config_detectron
from libs.detector.YOLOv3 import YOLOv3
from libs.detector.detectron import DETECTRON

# __all__ = ['build_detector_yolov3']

def build_detector_v3(cfg):
    net, layer_names, output_layers = config_yolov3(cfg)
    file_classes = cfg.YOLOV3.CLASS_NAMES

    return YOLOv3(net, file_classes, layer_names, output_layers)

def build_detector_detectron(cfg):
    predictor = config_detectron(cfg)
    file_classes = cfg.DETECTRON.CLASS_NAMES

    return DETECTRON(predictor, file_classes)
