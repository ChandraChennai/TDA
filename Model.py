import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F
from torch.optim import LBFGS

class Block(nn.Module):
    
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1) -> None:
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        
    def forward(self, x) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x
    


class ResNet_18(nn.Module):
    
    def __init__(self, image_channels, num_classes) -> None:
        
        super(ResNet_18, self).__init__()
        self.in_channels = 32
        self.conv1 = nn.Conv2d(image_channels, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #resnet layers
        self.layer1 = self.__make_layer(32, 32, stride=1)
        self.layer2 = self.__make_layer(32, 64, stride=2)
        self.layer3 = self.__make_layer(64, 128, stride=2)
        self.layer4 = self.__make_layer(128, 256, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def __make_layer(self, in_channels, out_channels, stride) -> nn.Sequential:
        
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
            
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride), 
            Block(out_channels, out_channels)
        )
    
    def forward(self, x) -> torch.Tensor:
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc(x))

        return self.relu(self.fc2(x)) 
    
    def identity_downsample(self, in_channels, out_channels) -> nn.Sequential:
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(out_channels)
        )
    
class SLP(nn.Module):
    def __init__(self, inputSize: int, hiddenSize: int,outputSize: int, out= False):
        super().__init__()
        self.fc1 = nn.Linear(inputSize, hiddenSize)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hiddenSize, outputSize)

        self.out = out

    def forward(self, x):
        out = self.act(self.fc1(x))
        out = self.fc2(out)


        return out
    
 



class ImgVector2(nn.Module):
    def __init__(self, inputChannel: int, resnetSize: int, inputSize: int, slpSize: int, outputSize: int):
        super().__init__()
        self.resnet = ResNet_18(inputChannel, resnetSize)
        self.fc = nn.Linear(3 + resnetSize, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.drop = nn.Dropout(p=0.20)
        self.out = nn.Linear(32, outputSize)

    def forward(self, img, label):
        imgFeat = self.resnet(img)
        catFeat = torch.cat([imgFeat, label], dim=1)
        out = self.relu(self.fc(catFeat))
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.out(out)

        return out

        
