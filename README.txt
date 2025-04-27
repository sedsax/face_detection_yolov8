Face Detection Using Yolov8

Dataset: https://universe.roboflow.com/mohamed-traore-2ekkp/face-detection-mik1i/dataset/27/download/yolov8

datasets/
└── face/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/

--------------------------------
Klasörlemeden sonra proje dizininde terminale sırasıyla yazılacak komutlar

- yolo_env\Scripts\activate

- pip install ultralytics

- yolo detect train data="C:/Users/Seda/Desktop/yolo_hazirlik/data.yaml" model=yolov8n.pt epochs=20 imgsz=640

----------------------------------------------------------------------------------------------------------------

20 epoch tamamlanınca eğitim bitecek ve çıktılar runs/detect/train2/ klasörüne kaydolacak.
Orada şunlar oluşacak:
weights/best.pt → En iyi model
weights/last.pt → Son epoch sonrası model
results.png → Loss ve accuracy grafiklerin

----------------------------------------------------------------------------------------------------------------
istenilen bir fotoğrafı test etmek için gerekli komut

- yolo task=detect mode=predict model=runs/detect/train2/weights/best.pt source="C:/Users/Seda/Desktop/yolo_hazirlik/datasets/face/images/val/Movie-on-2-18-25-at-8_25-PM_mov-0001_jpg.rf.d811931b759599f226517822348e3cc4.jpg"