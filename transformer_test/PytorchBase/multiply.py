import torch

shape1 = (2, 3)
shape2 = (3, 1)

# m1 = torch.randn(shape1, dtype=torch.float32)
# m2 = torch.randn(shape2, dtype=torch.float32)
m1 = torch.randint(0, 10, (2, 3))
m2 = torch.randint(0, 10, (3, 1))

matrix_product = m1 @ m2
matrix_product = m1 @ m2

print(m1)
print(m2)
print(matrix_product)
print(matrix_product.shape)
