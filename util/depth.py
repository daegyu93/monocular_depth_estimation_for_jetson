from __future__ import annotations
import time
import cv2
import numpy as np

import tensorrt as trt
# import pycuda.autoinit # Don't remove this line
import pycuda.autoprimaryctx
import pycuda.driver as cuda
from torchvision.transforms import Compose
from util.transform import Resize, NormalizeImage, PrepareForNet


class DepthEngine:
    def __init__(self, input_size, trt_engine_path):
        self.width = input_size # width of the input tensor
        self.height = input_size # height of the input tensor
        
        # self.cuda_device = cuda.Device(0)  # Assuming using GPU 0
        # self.cuda_context = self.cuda_device.make_context()

        self.runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR)) 
        self.engine = self.runtime.deserialize_cuda_engine(open(trt_engine_path, 'rb').read())
        self.context = self.engine.create_execution_context()
        print(f"Engine loaded from {trt_engine_path}")
        
        # Allocate pagelocked memory
        self.h_input = cuda.pagelocked_empty(trt.volume((1, 3, self.width, self.height)), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(trt.volume((1, 1, self.width, self.height)), dtype=np.float32)
        
        # Allocate device memory
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        
        # Create a cuda stream
        self.cuda_stream = cuda.Stream()
        
        # Transform functions
        self.transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=False,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image
        """
        image = image.astype(np.float32)
        image /= 255.0
        image = self.transform({'image': image})['image']
        image = image[None]
        
        return image
            
    def infer(self, image: np.ndarray) -> np.ndarray:
        """
        Infer depth from an image using TensorRT
        """
        # Preprocess the image
        image = self.preprocess(image)
        
        # t0 = time.time()
        
        # Copy the input image to the pagelocked memory
        np.copyto(self.h_input, image.ravel())
        
        # Copy the input to the GPU, execute the inference, and copy the output back to the CPU
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.cuda_stream)
        self.context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.cuda_stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.cuda_stream)
        self.cuda_stream.synchronize()
        
        depth = np.reshape(self.h_output, (self.width, self.height))
        return depth
    
    def __del__(self):
        #self.cuda_context.pop()
        #self.cuda_context.detach()
        #self.context.destroy()
        self.engine.destroy()
        self.cuda_stream.destroy()
        self.d_input.free()
        self.d_output.free()
        cuda.close()
        print("Engine destroyed")

            
if __name__ == '__main__':
    depth = DepthEngine()
    depth.run()
