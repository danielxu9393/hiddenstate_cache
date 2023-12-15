import torch

a = torch.randn(1, 5, 3)
b = a[:, -1, :]
print(b.shape)
c = b.unsqueeze(1)
print(c.shape)
