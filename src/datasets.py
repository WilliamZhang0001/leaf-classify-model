# 统一封装 transforms / dataset 加载
# src/datasets.py

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from src.transforms import train_transforms, val_test_transforms

def get_dataloaders(data_dir="data/split", batch_size=64, num_workers=4):
    train_ds = ImageFolder(f"{data_dir}/train", transform=train_transforms)
    val_ds   = ImageFolder(f"{data_dir}/val",   transform=val_test_transforms)
    test_ds  = ImageFolder(f"{data_dir}/test",  transform=val_test_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
