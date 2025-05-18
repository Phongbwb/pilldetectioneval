from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader

def get_dataloaders(config):
    
    train_dataset = VOCDetection(root='path/to/train', year='2007', image_set='train', download=False)
    val_dataset = VOCDetection(root='path/to/val', year='2007', image_set='val', download=False)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    return train_loader, val_loader