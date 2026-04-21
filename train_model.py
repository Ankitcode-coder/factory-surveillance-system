from ultralytics import YOLO

def run_training():
    model = YOLO("yolov8n.pt")
    model.train(
        data=r"C:\Users\Ankit kumar\Downloads\model\VisionSafe_Dataset\data.yaml",
        epochs=5,
        imgsz=640,
        batch=4,
        device=0,
        workers=2
    )

if __name__ == "__main__":  # 🔥 required on Windows
    run_training()
