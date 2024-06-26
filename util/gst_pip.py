import cv2
import os

os.environ['DISPLAY'] = ':0'

class GstCV2:
    def __init__(self, device_num, dewarp = True):
        
        if dewarp == True:
            gst_str = (
                f"nvv4l2camerasrc device=/dev/video{device_num} !"
                "video/x-raw(memory:NVMM),format=(string)UYVY,width=(int)1920,height=(int)1080,framerate=(fraction)30/1 !"
                f"queue ! nvvidconv ! nvdewarper config-file=/home/avs200/work/dghwang/config/config_{device_num}.txt ! nvvideoconvert ! "
                "video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink"     
            )
        else:
            gst_str = (
                f"nvv4l2camerasrc device=/dev/video{device_num} !"
                "video/x-raw(memory:NVMM),format=(string)UYVY,width=(int)1920,height=(int)1080,framerate=(fraction)30/1 !"
                "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink"
            )

        print(gst_str)
        self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

    def isOpened(self):
        return self.cap.isOpened()
    
    def read(self):
        ret, frame = self.cap.read()
        return ret, frame
    
    def release(self):
        self.cap.release()


if __name__ == '__main__':
    cap = GstCV2(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('frame', frame)
        
        key = cv2.waitKey(1)
        if key == 27:
            break
        
    cap.release()
    cv2.destroyAllWindows()
