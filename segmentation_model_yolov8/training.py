from ultralytics import YOLO
import torch

# Change if using M1 or other.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load a pre-trained YOLOv8 segmentation model.
model = YOLO("yolov8n-seg.pt")
model.to(device)

# Train the model. (Change Directory)
model.train(data="/home/mathew-macbook/development/EE577_Final_Project_Files/dataset_config.yaml", epochs=10, imgsz=256)
