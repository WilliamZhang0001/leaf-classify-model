import os, shutil
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm

# 原始数据路径（每类一个子文件夹）
DATASET_DIR = "E:/Learning/UNSW/Term2/9444/group_project/data/Plant_leave_diseases_dataset_with_augmentation"

# 输出目录
OUTPUT_DIR = "E:/Learning/UNSW/Term2/9444/group_project/data/split"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 获取图像路径和标签
image_paths = []
labels = []

for cls in os.listdir(DATASET_DIR):
    cls_path = os.path.join(DATASET_DIR, cls)
    if os.path.isdir(cls_path):
        for img_file in os.listdir(cls_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(cls_path, img_file))
                labels.append(cls)

# 第一次分割：train 70%，剩下30%再分 val/test
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths, labels, test_size=0.3, stratify=labels, random_state=42)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)

# 将图像移动到新的结构中
def move_files(paths, labels, split_name):
    for path, label in tqdm(zip(paths, labels), desc=f"Moving {split_name}"):
        dest_dir = os.path.join(OUTPUT_DIR, split_name, label)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(path, os.path.join(dest_dir, os.path.basename(path)))

move_files(train_paths, train_labels, "train")
move_files(val_paths, val_labels, "val")
move_files(test_paths, test_labels, "test")

print("✅ 数据划分完成")
