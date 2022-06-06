
import torch
import torch.nn as nn

import math

import torch
import torch.nn as nn
import torch.nn.functional as F




class conv_3d(nn.Module):
    def __init__(self):
        super(conv_3d, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        

        self._features = nn.Sequential(
            self.conv1,
            self.conv2
        )


    def forward(self, x):
        return self._features(x)# batch size, num channels,frames,  height, width




class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class DenseNet3(nn.Module):
    def __init__(self, depth=10, num_classes=60, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.4):
        super(DenseNet3, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n/2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(128, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        
        out = self.trans1(self.block1(out))
        
        out = self.trans2(self.block2(out))
        
        out = self.block3(out)
        
        out = self.relu(self.bn1(out))
        
        out = F.avg_pool2d(out, 3)
        
        
        return out
        
        


def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t
  
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        
        x = x.permute(0,1,3,4,2)# (batch, channels, height, width, frames)
        tList = [flatten(self.module(m)) for m in torch.unbind(x, dim=4) ]
        y = torch.stack(tList, dim=0)
        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1,x.size(4))  # (samples, output-size, timesteps) for conv1d
            
        else:
            y = y.view(x.size(4), x.size(0), -1)  # (timesteps, samples, output_size)
            
        return y
    

class block(nn.Module):
    def __init__(self,ni):
        super(block, self).__init__()
        self.conv1 = nn.Conv1d(ni, ni, 1)
        self.conv2 = nn.Conv1d(ni, ni, 3, 1, 1)

    def forward(self,x):
        residual = x
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out += residual
        
        return out
    
class seq_net(nn.Module):
    def __init__(self, input_size, hidden_size1):
        super(seq_net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size1

        self.c1 = block(input_size)
        self.p1 = nn.AvgPool2d(2)
        self.c2 = block(hidden_size1)
        self.p2 = nn.AvgPool1d(2)
        

    def forward(self, inputs):
      
        #print(inputs.shape)

        # Run through Conv1d and Pool1d layers
        c = self.c1(inputs)
        p = self.p1(c)
        c = self.c2(p)
        p = self.p2(c)
        out = F.relu(p)
        out = out.view(out.size(0),-1)
        return out




# or we can use lstm at last



#   class seq_net(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(seq_net, self).__init__()
#         self.hidden_size = hidden_size
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

#     def forward(self, x):
#         #print(inputs.shape)
#         # Run through LSTM layer
#         out, _ = self.lstm(x)
#         out = out[:,-1,:]
#         out = out.view(x.size(0),-1)
#         return out