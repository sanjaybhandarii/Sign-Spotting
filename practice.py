import torch
import torch.nn as nn

x = torch.randn(1, 3,5, 64, 64)
x = torch.tensor(torch.unbind(x, dim=4))


y = nn.Conv2d(3, 64, kernel_size=5, padding=1)(x)
print(y.shape)