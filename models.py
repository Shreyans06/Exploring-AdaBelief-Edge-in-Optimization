import torch
import torch.nn as nn
import operator
import functools


class VGG(nn.Module):
    '''
    VGG model
    '''

    def __init__(self, architecture, num_classes=10, input_dims=[3, 32, 32]):
        super(VGG, self).__init__()
        self.architecture = architecture
        self.num_classes = num_classes
        self.input_dims = input_dims
        self.convs = self.conv()
        self.fcs = self.fc()

    def conv(self):
        layers = []
        input_channels = self.input_dims[0]

        for value in self.architecture:
            if type(value) == int:
                layers += [nn.Conv2d(in_channels=input_channels, out_channels=value, kernel_size=3, padding=1),
                           nn.BatchNorm2d(value), nn.ReLU(inplace=True)]
                input_channels = value
            else:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)

    def fc(self):

        features_size = functools.reduce(operator.mul, list(self.convs(torch.rand(1, *self.input_dims)).shape))

        return nn.Sequential(nn.Linear(features_size, 512),
                             nn.ReLU(inplace=True),
                             nn.Dropout(p=0.5),
                             nn.Linear(512, 512),
                             nn.ReLU(inplace=True),
                             nn.Dropout(p=0.5),
                             nn.Linear(512, self.num_classes)
                             )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.fcs(x)
        return x
