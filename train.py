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
    ('features', TimeDistributed(DenseNet3())),
    ('backend', seq_net(input_size=6048, hidden_size1=3024)),
    ('fc', nn.Sequential( nn.Dropout(p=0.5), nn.Linear(3024, 60)))
]))





# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()
# specify optimizer and learning rate
optimizer = optim.SGD(
  [
        {"params": model.fc.parameters(), "lr": 1e-3},
        {"params": model.features.parameters(), "lr": 1e-5},
        {"params": model.backend.parameters(), "lr": 1e-5},
        {"params": model.frontend.parameters(), "lr": 1e-4},
  ],
  momentum = 0.9
)
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# state = torch.load("../input/sign-classification/model_optimizer.pt")
# model.load_state_dict(state['model_state_dict'])
model.half()
# model.cuda()

#optimizer.load_state_dict(state['optimizer_state_dict'])

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    correct = 0
    num_samples = 0
    
    for epoch in tqdm(range(epoch)):
        for batch_idx, (data, target) in enumerate(train_loader):
            
            data = data.to(device)
            target = torch.tensor(target).to(device)

    
    
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()* data.size(0)
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            num_samples += pred.shape[0]
            print(num_samples)
            correct += pred.eq(target.view_as(pred)).sum().item()
            print("correct till now:",correct)
        
    train_loss /= num_samples
    print('Epoch: {} , Training Accuracy: {}/{} ({:.0f}%) Training Loss: {:.6f}'.format(
                epoch, correct, num_samples,
                100. * correct / num_samples, train_loss))


dataloader = load_dataloader(batch_size=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
train(model,device,dataloader,optimizer,criterion,1)