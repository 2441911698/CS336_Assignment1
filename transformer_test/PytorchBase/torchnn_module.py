import torch
import torch.nn as nn
import torch.optim as optim

# 1. 准备模拟数据
X = torch.randn(100, 1)
y_true = 2.0 * X + 1.0 + torch.randn(100, 1) * 0.1

# 2. 定义模型、损失函数和优化器
model = nn.Linear(in_features=1, out_features=1)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ==========================================
# THE FINAL, CLEAN TRAINING LOOP
# ==========================================

epochs = 100  # [cite: 4]

for epoch in range(epochs):
    ### FORWARD PASS ###
    y_hat = model(X)

    ### CALCULATE LOSS ###
    loss = loss_fn(y_hat, y_true)

    # THE THREE-LINE MANTRA #
    # 1.Zero the gradients
    optimizer.zero_grad()
    # 2.Compute gradients
    loss.backward()
    # 3.Update the parameters
    optimizer.step()

    # Optional: Print progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch:02d}: Loss={loss.item():.4f}")