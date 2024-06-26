from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(
    format="engine",
    dynamic=True,  
    batch=8,  
    workspace=4,  
    int8=True,
    data="coco.yaml",  
)

# Load the exported TensorRT INT8 model
model = YOLO("yolov8n.engine", task="detect")