import torch

data = [[1,2,3],[4,5,6]]
my_tensor = torch.tensor(data)

shape = (2,3)

ones = torch.ones(shape)# print(f"oneTensor:\n{ones}")
zeros = torch.zeros(shape)# print(f"zeroTensor:\n{zeros}")
random = torch.randn(shape)

rand_like = torch.rand_like(my_tensor,dtype=torch.float)

# print(f"Template Tensor:\n{rand_like}\n")
# print(f"Random Tensor:\n{random}\n")
print(f"Shape:{random.shape}")
print(f"Shape:{random.dtype}")
print(f"Shape:{random.device}")