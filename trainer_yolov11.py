from ultralytics import YOLO
import argparse

def train_yolov11(data_path, epochs, imgsz, batch, name):
    model = YOLO('yolov11.pt')  # hoặc sử dụng .yaml nếu training from scratch

    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=name,
        project='runs/train',
        save=True,
        save_period=1
    )

    print(" Training YOLOv11 completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", type=str, default="yolov11_run")
    args = parser.parse_args()

    train_yolov11(args.data, args.epochs, args.imgsz, args.batch, args.name)