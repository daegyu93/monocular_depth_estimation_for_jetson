import cv2
import numpy as np
import vpi
import os

os.environ['DISPLAY'] = ':0'

class Dewarp:
    def __init__(self, width, height, cal_path):
        self.cal_path = cal_path    
        frameSize = (width, height)
        grid = vpi.WarpGrid(frameSize)

        # get paramter
        camMatrix = np.eye(3)
        X = np.eye(3,4)
        coeffs = np.zeros((4,))

        with np.load(self.cal_path) as X:
            camMatrix, X, coeffs = [X[i] for i in ('arr_0', 'arr_1', 'arr_2')]
        self.undist_map_0 = vpi.WarpMap.fisheye_correction(grid, K=camMatrix, X=X, coeffs=coeffs, mapping=vpi.FisheyeMapping.EQUIDISTANT)

    def distortion_free(self, frame):
        with vpi.Backend.CUDA:
            imgCorrected_0 = vpi.asimage(frame).convert(vpi.Format.NV12_ER).remap(self.undist_map_0, interp=vpi.Interp.CATMULL_ROM).convert(vpi.Format.RGB8)
            frame = imgCorrected_0.cpu()
        
        return frame
