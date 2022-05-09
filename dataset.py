import torch
import numpy as np
from torch.utils.data import (
    Dataset,
    DataLoader,
) 
import pickle

import os



class SignDataset(Dataset):
    def __init__(self,x,y):
        self.x = x
        print("len x" , len(self.x))
        self.y = y
        print("len y" , len(self.y))


    def __len__(self):

        return len(self.y)

        # length = 0
        # with open(self.file, 'rb') as f:
        #     data = pickle.load(f)

        # for x in data:
        #     length += len(data[x])
        # return length


    def __getitem__(self, index):
        return self.x[index], self.y[index]
        