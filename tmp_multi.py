import os
import cv2

from util.yolo import YOLOV8
from util.depth import DepthEngine
from util.gst_pip import GstCV2
import numpy as np
import time

import queue
import threading

import copy


os.environ["DISPLAY"] = ":0"

def post_process(depth_raw):
    # depth = (depth_raw * 10).astype(np.uint8)
    #nomalize
    depth = (depth_raw - np.min(depth_raw)) / (np.max(depth_raw) - np.min(depth_raw)) * 255
    depth = depth.astype(np.uint8)

    color_map = cv2.COLORMAP_JET
    depth_rgb = cv2.applyColorMap(depth, color_map)
    depth_rgb = cv2.resize(depth_rgb, (960, 540))
    return depth_rgb

def inference_thread(q, index, yolo, depth, frame, enable = True):
    if enable == False:
        q.put((index, frame))
        return
    
    boxes = yolo.infer(frame)
    depth_raw = depth.infer(frame)
    frame = copy.copy(post_process(depth_raw))

    for box in boxes:
        if box.cls != 0:
            continue
        x1, y1, x2, y2 = box.cpu().xyxy[0]
        x1 = (int)(x1.item())
        y1 = (int)(y1.item())
        x2 = (int)(x2.item())
        y2 = (int)(y2.item())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 4)
    q.put((index, frame))

def main():
    trt_engine_path = '/home/avs200/work/dghwang/weights/depth_anything_vits14_364.trt'
    yolov8_path = '/home/avs200/work/dghwang/weights/yolov8n_int8.engine'
    input_size = 364

    yolo1 = YOLOV8(yolov8_path)
    yolo2 = YOLOV8(yolov8_path)
    yolo3 = None #YOLOV8(yolov8_path)
    yolo4 = YOLOV8(yolov8_path)
    depth1 = DepthEngine(input_size, trt_engine_path)
    depth2 = DepthEngine(input_size, trt_engine_path)
    depth3 = None #DepthEngine(input_size, trt_engine_path)
    depth4 = DepthEngine(input_size, trt_engine_path)
    
    cap1 = GstCV2(0)
    cap2 = GstCV2(1)
    cap3 = GstCV2(2, dewarp = False)
    cap4 = GstCV2(3)

    caps = [cap1, cap2, cap3, cap4]
    yolos = [yolo1, yolo2, yolo3, yolo4]
    depths = [depth1, depth2, depth3, depth4]

    result_queue = queue.Queue()
    while cap1.isOpened():
        start_time = time.time()
        threads = []

        for i in range(4):
            _, frame = caps[i].read()
            frame = cv2.resize(frame, (960, 540))
            if i == 2:
                threads.append(threading.Thread(target=inference_thread, args=(result_queue, i, yolos[i], depths[i], frame, False)))
            else :
                threads.append(threading.Thread(target=inference_thread, args=(result_queue, i, yolos[i], depths[i], frame)))
            threads[i].start()

        for i in range(4):
            threads[i].join()

        frames = {}
        for i in range(4):
            index, frame = result_queue.get()
            frames[index] = frame

        # show 2by 2 frame
        frame1 = np.hstack((frames[0], frames[1]))
        frame2 = np.hstack((frames[2], frames[3]))
        frame = np.vstack((frame1, frame2))
        cv2.imshow('frame', frame)
        # print(f'inference time: {time.time() - start_time:.2f}')
        #print(f'FPS: {1 / (time.time() - start_time):.2f}')
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap1.release()
        
if __name__ == '__main__':
    main()
    
