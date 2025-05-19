import os
import torch
from models.faster_rcnn import get_faster_rcnn_model
from models.retinanet import get_retinanet_model
from models.ssd import get_ssd_model
from utils import get_dataloaders

# External trainers
from trainer_yolov11 import train_yolov11
from trainer_rtdetr import train_rtdetr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(model, model_name):
    os.makedirs("checkpoints", exist_ok=True)
    path = f"checkpoints/{model_name}.pth"
    torch.save(model.state_dict(), path)
    print(f" Saved checkpoint: {path}")

def fine_tune(model_name, config):
    if model_name == "yolov11":
        train_yolov11(config)
        return

    elif model_name == "rt_detr":
        train_rtdetr(config)
        return

    # Các mô hình còn lại dùng PyTorch thông thường
    train_loader, val_loader = get_dataloaders(config)

    if model_name == "faster_rcnn":
        model = get_faster_rcnn_model(num_classes=config["num_classes"], pretrained=True)
    elif model_name == "retinanet":
        model = get_retinanet_model(num_classes=config["num_classes"], pretrained=True)
    elif model_name == "ssd":
        model = get_ssd_model(num_classes=config["num_classes"], pretrained=True)
    else:
        raise ValueError("Unsupported model")

    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    
    model.train()
    for epoch in range(config["epochs"]):
        total_loss = 0
        for images, targets in train_loader:
            images = images.to(DEVICE)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            output = model(images, targets)
            loss = sum(loss for loss in output.values())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[{model_name}] Epoch {epoch+1}/{config['epochs']} - Loss: {total_loss:.4f}")

    save_checkpoint(model, model_name)

def main():
    config = {
        "num_classes": 19,
        "batch_size": 16,
        "lr": 0.0005,
        "epochs": 10,
        "img_size": 416,
        "train_path": "datasets/yolo/train/images",
        "val_path": "datasets/yolo/val/images",
        "data_yaml": "datasets/yolo/data.yaml",
        "output_dir": "checkpoints"
    }

    for model_name in ["yolov11", "faster_rcnn", "retinanet", "ssd", "rt_detr"]:
        print(f"\n Fine-tuning {model_name.upper()}...")
        fine_tune(model_name, config)

if __name__ == "__main__":
    main()
