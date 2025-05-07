import cv2
import os
import matplotlib.pyplot as plt
if __name__ == '__main__':
  
    from ultralytics import YOLO
    labels_folder = r"C:/Users/Seda/Desktop/yolo_hazirlik/datasets/face/labels"
    outputs_folder = r"C:/Users/Seda/Desktop/yolo_hazirlik/datasets/face/images/train"

    # Eğitim sonrası ilgili weight dosyaasını 
    model2 = YOLO(r'C:/Users/Seda/Desktop/yolo_hazirlik/runs/detect/train17/weights/best.pt')
    results = model2.val(data=r'C:/Users/Seda/Desktop/yolo_hazirlik/data.yaml',iou=0.4, source=r'C:/Users/Seda/Desktop/yolo_hazirlik/datasets/face/images/val', imgsz=640, save=True)
 