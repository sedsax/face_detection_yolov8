from ultralytics import YOLO
import torch
import argparse
import os

def train_model(model_path, data_path, epochs, imgsz, device, save_example_image=None):
    try:
        if not os.path.exists(model_path):
            print(f"Model dosyası bulunamadı: {model_path}")
            return
        if not os.path.exists(data_path):
            print(f"Veri yaml dosyası bulunamadı: {data_path}")
            return
        print("CUDA kullanılabilir mi?:", torch.cuda.is_available())
        print("Kullanılan aygıt:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
        model = YOLO(model_path)
        results = model.train(data=data_path, epochs=epochs, device=device, imgsz=imgsz)
        best_weight_path = results.save_dir / 'weights' / 'best.pt'
        print(f"En iyi model kaydedildi: {best_weight_path}")
        print(f"Eğitim tamamlandı. Sonuçlar: {results.metrics}")
        if save_example_image:
            if not os.path.exists(save_example_image):
                print(f"Örnek resim bulunamadı: {save_example_image}")
            else:
                print(f"Eğitim sonrası örnek tahmin yapılıyor: {save_example_image}")
                model = YOLO(str(best_weight_path))
                pred = model(save_example_image, device=device)
                print(f"Örnek tahmin sonucu: {pred}")
        return results
    except Exception as e:
        print(f"Eğitim sırasında hata oluştu: {e}")

def predict_image(model_path, image_path, device):
    try:
        if not os.path.exists(model_path):
            print(f"Model dosyası bulunamadı: {model_path}")
            return
        if not os.path.exists(image_path):
            print(f"Resim dosyası bulunamadı: {image_path}")
            return
        model = YOLO(model_path)
        results = model(image_path, device=device)
        print(f"Tahmin sonuçları: {results}")
        return results
    except Exception as e:
        print(f"Tahmin sırasında hata oluştu: {e}")

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Eğitim ve Tahmin Scripti")
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='train', help='Çalışma modu: train veya predict')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Model dosya yolu')
    parser.add_argument('--data', type=str, default='data.yaml', help='Veri yaml dosya yolu (sadece eğitim için)')
    parser.add_argument('--epochs', type=int, default=50, help='Epoch sayısı (sadece eğitim için)')
    parser.add_argument('--imgsz', type=int, default=224, help='Görüntü boyutu (sadece eğitim için)')
    parser.add_argument('--device', type=str, default='0', help='Kullanılacak cihaz (0, cpu, cuda:0 vs)')
    parser.add_argument('--image', type=str, help='Tahmin yapılacak resim dosya yolu (sadece predict için)')
    parser.add_argument('--example', type=str, help='Eğitim sonrası örnek tahmin için resim yolu (sadece train için)')
    args = parser.parse_args()
    if args.mode == 'train':
        train_model(args.model, args.data, args.epochs, args.imgsz, args.device, save_example_image=args.example)
    elif args.mode == 'predict':
        if not args.image:
            print('Predict modu için --image parametresi gereklidir!')
        else:
            predict_image(args.model, args.image, args.device)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Windows için güvenlik önlemi
    main()
