import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


# ==========================================
# 1. DataLoader: 处理数据 (Dataset + DataLoader)
# ==========================================
class MyDataset(Dataset):
    def __init__(self):
        # 模拟 1000 个样本，每个样本 20 个特征
        self.data = torch.randn(1000, 20)
        self.labels = torch.randint(0, 2, (1000,))  # 0 或 1 的二分类标签

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


dataset = MyDataset()
# DataLoader 负责分批次 (batching)、打乱 (shuffling) 和多线程加载
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# ==========================================
# 2. nn.Module: 构建网络
# ==========================================
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()  # 必须调用父类初始化
        self.fc1 = nn.Linear(20, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        # 定义数据流向（前向传播）
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = SimpleNet()

# ==========================================
# 3. Optimizer: 更新参数 (结合 Loss)
# ==========================================
criterion = nn.CrossEntropyLoss()  # 损失函数
# 优化器需要知道要更新哪些参数 (model.parameters()) 以及学习率
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 4. 将三大件结合：标准训练循环 (Training Loop)
# ==========================================
epochs = 500
for epoch in range(epochs):
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # 步骤 1: 清空过去的梯度
        optimizer.zero_grad()

        # 步骤 2: 前向传播，计算预测值
        outputs = model(inputs)

        # 步骤 3: 计算损失 (Loss)
        loss = criterion(outputs, targets)

        # 步骤 4: 反向传播，计算当前梯度
        loss.backward()

        # 步骤 5: 优化器根据梯度更新网络参数
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")