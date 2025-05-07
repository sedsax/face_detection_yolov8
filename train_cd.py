import cv2
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    from ultralytics import YOLO

    # Eğitim için bu 2 satır yeterli
    model = YOLO("yolov8s.pt")
    results = model.train(
        data="data.yaml",
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        epochs=100,
        batch=4,
        imgsz=640
    )