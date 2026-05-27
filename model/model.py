import torch 
from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0=nn.Conv2d(
            in_channels=3,
            out_channels=16,
            stride=1,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.bn0=nn.BatchNorm2d(num_features=16)
        self.maxpool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv1=nn.Conv2d(
            in_channels=16,
            out_channels=32,
            stride=1,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.bn1=nn.BatchNorm2d(num_features=32)
        self.dropout=nn.Dropout(p=0.2)
        self.conv2=nn.Conv2d(
            in_channels=32,
            out_channels=64,
            stride=1,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.bn2=nn.BatchNorm2d(num_features=64)
        self.fc0=nn.Linear(
            in_features=64*28*28,
            out_features=64,
        )
        self.fc1=nn.Linear(
            in_features=64,
            out_features=32,
        )
        self.fc2=nn.Linear(
            in_features=32,
            out_features=16,
        )
        self.fc3=nn.Linear(
            in_features=16,
            out_features=1,
        )
        
    def forward(self,x):
        x=nn.functional.relu(self.bn0(self.conv0(x)))
        x=self.maxpool(x)
        x=nn.functional.relu(self.bn1(self.conv1(x)))
        x=self.maxpool(x)
        x=nn.functional.relu(self.bn2(self.conv2(x)))
        x=self.maxpool(x)
        x=x.reshape(x.shape[0],-1)
        x=nn.functional.relu(self.fc0(x))
        x=self.dropout(x)
        x=nn.functional.relu(self.fc1(x))
        x=nn.functional.relu(self.fc2(x))
        x=nn.functional.sigmoid(self.fc3(x))
        return x