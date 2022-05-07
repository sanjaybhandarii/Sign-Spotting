import imp


import torch
a = torch.rand(1,3,256,256)
b=torch.std_mean(a, unbiased=False)
print(b)