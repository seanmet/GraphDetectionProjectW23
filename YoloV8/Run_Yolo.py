from PIL import Image, ImageDraw
import PIL
from ultralytics import  YOLO
import os

if not os.path.exists('YOLO_FINAL_RUNS/complex_not_pretrained'):
    os.makedirs('YOLO_FINAL_RUNS/complex_not_pretrained')
    
pwd = os.getcwd()
print("Current Working Directory " , pwd)
model = YOLO('yolov8n.pt')
train_params = {
    'data': os.path.join(os.getcwd(), 'data.yaml'),
    'epochs': 75,
    'batch': 82,
    'imgsz': 640,
    'pretrained': False,
    'plots': True,
    'project': 'YOLO_FINAL_RUNS/complex_not_pretrained',
    # Add other training parameters here as needed
}



# Train the model with the specified parameters
results = model.train(**train_params)

# create a new directory to save the model name:models


model.save('YOLO_FINAL_RUNS/complex_not_pretrained')
