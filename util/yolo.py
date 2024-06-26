import cv2
import os
from ultralytics import YOLO
import time

import torch
torch.cuda.empty_cache()

os.environ['DISPLAY'] = ':0'

class YOLOV8:
    def __init__(self, engine_path):
        self.model = YOLO(engine_path, verbose=False)

    def infer(self, frame):
        results = self.model(frame, conf=0.5, iou=0.5, verbose=False)
        return results[0].boxes

