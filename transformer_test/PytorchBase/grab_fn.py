import torch

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
x = torch.tensor(4.0, requires_grad=True)

y = a + b #First operation

z = x * y #Second operation

print(f"grab_fn for z:{z.grad_fn}")
print(f"grab_fn for Y:{y.grad_fn}")
print(f"grab_fn for A:{a.grad_fn}") #计算图