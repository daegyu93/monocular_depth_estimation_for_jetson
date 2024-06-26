import os
import cv2

from util.yolo import YOLOV8
from util.depth import DepthEngine
from util.gst_pip import GstCV2
import numpy as np
import time

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VERBOSE_LOGGING'] = '1'

os.environ["DISPLAY"] = ":0"

def post_process(depth_raw):
    depth = (depth_raw * 10).astype(np.uint8)
    depth_rgb = cv2.applyColorMap(depth, cv2.COLORMAP_BONE)
    depth_rgb = cv2.resize(depth_rgb, (1920, 1080))
    return depth_rgb

def main():
    # trt_engine_path = 'weights/depth_anything_vits14_364.trt'
    trt_engine_path = 'weights/metric.trt'
    input_size = 518

    yolo = YOLOV8('weights/yolov8n_int8.engine')
    depth = DepthEngine(input_size, trt_engine_path)
    
    cap1 = GstCV2(0)

    while cap1.isOpened():
        start_time = time.time()
        ret, frame = cap1.read()
        depth_raw = depth.infer(frame)
        frame = post_process(depth_raw)
        print(f'FPS: {1 / (time.time() - start_time):.2f}')
        cv2.imshow('frame', frame)
        
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap1.release()
        
if __name__ == '__main__':
    main()
    