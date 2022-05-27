
import torch
from torch.nn.utils.rnn import pad_sequence
a = torch.randn(5,3,7,8,8)


# c = torch.randn(2,3,7,8,8)

# print("a:",a)

# print("\n c:",c)
# d = pad_sequence([a,c],batch_first=True)
# a,c = [m for m in torch.unbind(d, dim=0)]
# print("\n a:",a)

# print("\n c:",c)


def pad_constant(tensor, length, value):
    return torch.cat([tensor, tensor.new_zeros(length - tensor.size(0), *tensor.size()[1:])], dim=0)

f = pad_constant(a,41,0)

print(f.shape)