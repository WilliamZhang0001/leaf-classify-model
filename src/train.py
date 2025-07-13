# 模型训练主脚本
from src.datasets import get_dataloaders

train_loader, val_loader, _ = get_dataloaders(batch_size=64)

for epoch in range(num_epochs):
    train_one_epoch(model, train_loader)
    validate(model, val_loader)