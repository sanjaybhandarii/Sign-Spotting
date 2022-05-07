

import torch
x = torch.randn(1,2,2)
print(x)
print(x.permute(0,2,1))