版本环境：
GPU device name: NVIDIA GeForce RTX 4080 Laptop GPU
python: 3.12
CUDA : 12.8
PyTorch : 2.7.1+cu128 (GPU版本)
cuDNN : v8.9.7.29
GPU Driver : 576.88

项目文件简介：
```
group_project/
├─ data/                 # 原始 & 处理后数据
│  ├─ Plant_leave_diseases_dataset_with_augmentation   # 原始增强数据
│  └─ split_with_713     # 处理后数据
│     ├─ test
│     ├─ train
│     └─ val
├─ notebooks/            # EDA + 训练记录
├─ src/
│  ├─ datasets.py        # 自定义 Dataset / transforms
│  ├─ models/            # baseline9.py, efficientnet.py...
│  ├─ train.py           # 可CLI化训练脚本
│  └─ infer.py           # 推理 & 可视化
├─ outputs/              # checkpoints, logs, tensorboard, 图片
└─ demo/
```
