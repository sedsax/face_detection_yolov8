import cv2
import os
import matplotlib.pyplot as plt
if __name__ == '__main__':
  
    from ultralytics import YOLO

    # Eğitim sonrası ilgili weight dosyaasını 
    model2 = YOLO(r'C:/Users/Seda/Desktop/yolo_hazirlik/runs/detect/train17/weights/best.pt')
    results = model2.predict(data=r'C:/Users/Seda/Desktop/yolo_hazirlik/data.yaml',iou=0.4, source=r'C:/Users/Seda/Desktop/yolo_hazirlik/datasets/face/images/train', imgsz=640, save=True)
   