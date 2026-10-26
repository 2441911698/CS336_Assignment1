import torch

# 生成用于训练的模拟数据 (为了匹配图中 W=2.000, b=1.000 的设定)
X = torch.randn(100, 1)
y_true = 2.0 * X + 1.0 + torch.randn(100, 1) * 0.1

# Hyperparameters (超参数)
learning_rate, epochs = 0.01, 100

# Re-initialize parameters (初始化参数)
W, b = torch.randn(1, 1, requires_grad=True), torch.randn(1, requires_grad=True)

# Training Loop (训练循环)
for epoch in range(epochs):
    # Forward pass and loss
    y_hat = X @ W + b
    loss = torch.mean((y_hat - y_true) ** 2)

    # Backward pass
    loss.backward()

    # Update parameters
    with torch.no_grad():
        W -= learning_rate * W.grad
        b -= learning_rate * b.grad

    # Zero gradients
    W.grad.zero_()
    b.grad.zero_()

    # 每 10 轮输出一次 loss 和参数信息
    if epoch % 10 == 0:  #
        print(f"Epoch {epoch:02d}: Loss={loss.item():.4f}, W={W.item():.3f}, b={b.item():.3f}")  # [cite: 3]

# 训练结束后输出最终参数与真实参数的对比
print(f"\nFinal Parameters: W={W.item():.3f}, b={b.item():.3f}")  # [cite: 3]
print(f"True Parameters:  W=2.000, b=1.000")  # [cite: 3]