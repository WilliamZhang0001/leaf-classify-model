#对训练/测试/验证数据集进行变换，以满足项目要求：该系统应能够处理由于不同的光照条件、叶片方向和其他环境因素导致的外观变化
import torch
from torchvision import transforms

# ─── 训练集 Data Augmentation ───
train_transforms = transforms.Compose([
    # 随机缩放并裁剪到 224×224，scale 保证裁剪区域至少占原图 80%
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    # 随机水平 / 垂直翻转
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # 颜色扰动：亮度、对比度、饱和度、色调
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.05),
    # 随机旋转 ±20°
    transforms.RandomRotation(20),
    # 转为 Tensor 并归一化到 ImageNet 统计
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─── 验证集 / 测试集（普通推理）────
val_test_transforms = transforms.Compose([
    # 先长边缩放到 256，再中心裁剪到 224
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─── Test-Time Augmentation (TTA) ───
# 以 TenCrop 为例：对每张图生成 10 个裁剪，然后对每个裁剪做 ToTensor+Normalize，最后堆叠成一个 batch
tta_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.TenCrop(224),  # 返回 tuple(PIL.Image) 长度=10
    transforms.Lambda(lambda crops: torch.stack([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])(
            transforms.ToTensor()(crop)
        ) for crop in crops
    ])),
])
