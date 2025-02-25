from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the trained YOLOv8 segmentation model. (Change Directory)
model = YOLO("/home/mathew-macbook/runs/segment/train/weights/best.pt")

# Path to an example MRI slice image. (Change Directory)
test_image_path = "/home/mathew-macbook/development/EE577_Final_Project_Files/data/images/train/GK.103_1_tumor1/GK.103_1_tumor1_slice_163.png"

# Run prediction.
results = model.predict(test_image_path, save=True)
