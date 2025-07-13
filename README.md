版本环境：
GPU device name: NVIDIA GeForce RTX 4080 Laptop GPU

python: 3.12

CUDA : 12.8

PyTorch : 2.7.1+cu128

cuDNN : v8.9.7.29

GPU Driver : 576.88

项目文件简介：
```
group_project/
├─ data/                 # 原始 & 处理后数据
│  ├─ Plant_leave_diseases_dataset_with_augmentation   # 原始增强数据
│  ├─split_dataset.py    # 数据处理
│  └─ split_with_713     # 处理后数据
│     ├─ test            # 测试集(每个类别150张照片)
│     ├─ train           # 训练集(每个类别700张照片)
│     └─ val             # 验证集(每个类别150张照片)
├─ notebooks/            # 训练记录+工作记录
├─ src/
│  ├─ datasets.py        # 统一封装 transforms / dataset 加载
│  ├─ models/            # 最终训练模型baseline9.py, efficientnet.py...
│  ├─ train.py           # 模型训练主脚本
│  ├─ eval.py            # 模型验证（val/test）脚本
│  ├─ transforms.py      # 对训练/测试/验证数据集进行变换，以满足项目要求
│  └─ inference.py       # 单张图推理（含 TenCrop数据增强策略)
├─ outputs/              # checkpoints, logs, tensorboard, 图片
└─ demo/
```
