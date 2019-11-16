## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self,num_layers=3,num_filters=16,kernel_size=(3,3),drop_prob1=0.2, 
                 drop_prob2 = 0.5,pool_layers=[0,1],drop_layers=[0,1]):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel

        # input: (1,224,224)
        in_channels = 1
        out_channels = num_filters
        dim = 224
        
        layers= []
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=1))
            layers.append(nn.ReLU())
            
            if i in drop_layers:
                layers.append(nn.Dropout2d(drop_prob1))
                
            layers.append(nn.BatchNorm2d(out_channels))
            
            in_channels = out_channels
            out_channels *= 2
            dim = (dim+2-kernel_size[0])+1
            
            if i in pool_layers:
                layers.append(nn.MaxPool2d(2,2))
                dim = dim//2
            
        
        # feature block
        self.features = nn.Sequential(*layers)
        
        # fc layers
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout1 = nn.Dropout(drop_prob2)
        self.fc1 = nn.Linear(in_channels,256)
        self.fc2 = nn.Linear(256,136)

        
    def forward(self, x):
        net = self.features(x)
        net = self.avgpool(net)
        
        net = net.view(net.size(0),-1)
        
        net = self.dropout1(net)
        net = F.relu(self.fc1(net))
        net = self.fc2(net)
        return net