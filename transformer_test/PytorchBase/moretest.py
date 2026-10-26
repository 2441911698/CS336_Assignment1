import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn as nn


# ==========================================
# 图像版 Dataset：如何把 jpg/png 变成 Tensor
# ==========================================
class MyImageDataset(Dataset):
    def __init__(self):
        # 实际项目中，这里通常是一个包含所有图片路径的列表
        # 例如: self.image_paths = ["cat1.jpg", "dog2.png", ...]
        self.num_samples = 100

        # 定义图像转换管道：核心是 ToTensor()
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # 把所有图片缩放到统一大小 64x64
            transforms.ToTensor()  # 关键步骤：转为张量并归一化
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. 模拟从硬盘读取一张 RGB 图像 (真实场景用 Image.open(path))
        # 此时它是普通的 PIL 图像，形状相当于 (H=64, W=64, C=3)，像素值 0~255
        fake_pixels = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        image = Image.fromarray(fake_pixels)

        # 2. 魔法发生在这里：传入 transform 管道
        # ToTensor() 会做两件事：
        #   - 轴转换：(H, W, C) -> (C, H, W) 即变成了 (3, 64, 64)
        #   - 数值归一化：将 0~255 的整数除以 255，变成 0.0~1.0 的浮点数
        img_tensor = self.transform(image)

        # 3. 生成标签
        label = torch.tensor(1)  # 假定类别为 1

        return img_tensor, label

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 * 64 * 64 = 12288 个特征
        self.fc1 = nn.Linear(12288, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        # 进网络前，先把 [32, 3, 64, 64] 压平为 [32, 12288]
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 测试一下张量的形状变化
dataset = MyImageDataset()
img_tensor, label = dataset[0]

print(f"单张图片转为 Tensor 后的形状: {img_tensor.shape}")
# 输出: torch.Size([3, 64, 64]) （3个颜色通道，高64，宽64）

# 当它进入 DataLoader 后，会自动增加 Batch 维度 (B, C, H, W)
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=32)

for batch_imgs, batch_labels in dataloader:
    print(f"一个 Batch 的图像张量形状: {batch_imgs.shape}")
    # 输出: torch.Size([32, 3, 64, 64]) （32张图，3通道，64x64）
    break