import torch
import torch.nn as nn

import math

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from model import *
from utils import *
import torch.optim as optim


n_classes = 60

model = nn.Sequential(OrderedDict([
    ('frontend', conv_3d()),
    ('features', TimeDistributed(DenseNet())),
    ('backend', SEQ(input_size=5000, hidden_size1=2500)),
    ('fc', nn.Sequential( nn.Dropout(p=0.5), nn.Linear(2500, n_classes) ))
]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()
# specify optimizer and learning rate
optimizer = optim.SGD(
  [
        {"params": model.fc.parameters(), "lr": 1e-3},
        {"params": model.backend.parameters(), "lr": 1e-5},
        {"params": model.features.parameters(), "lr": 1e-4},
        {"params": model.frontend.parameters(), "lr": 1e-4},
  ],
  momentum = 0.9
)


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    correct = 0
    
    train_steps = train_loader.get_sample_size()
    
    gen = train_loader.generator()
    
    for batch_idx in tqdm(range(train_steps)): 
        data, target = next(gen)
        data, target = data.to(device), torch.max(target.long().to(device), 1)[1]
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        
    train_loss /= train_loader.get_sample_size()
    print('Epoch: {} , Training Accuracy: {}/{} ({:.0f}%) Training Loss: {:.6f}'.format(
                epoch, correct, train_loader.get_sample_size(),
                100. * correct / train_loader.get_sample_size(), train_loss))


dataloader = load_dataloader(batch_size=8)

x,y = iter((dataloader)).next()
print(y)

#train(model,device,dataloader,optimizer,criterion,4)