from ultralytics import YOLO
import torch

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = YOLO("yolov8m-seg.pt")
    model.to(device)

    model.train(data="EE577_Final_Project_Files/2d_training_files/data_config.yaml", epochs=200, batch=-1, imgsz=512, save_period=1)