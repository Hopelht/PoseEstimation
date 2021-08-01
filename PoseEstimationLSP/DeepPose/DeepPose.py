#DeepPose模型

import torch
import torch.nn as nn


class DeepPose(nn.Module):
    def __init__(self):
        super(DeepPose, self).__init__()

        self.model = torch.nn.Sequential(

        nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),
        nn.ReLU(inplace=True),
        nn.LocalResponseNorm(size=2, alpha=2e-05, beta=0.75, k=1),  #局部响应归一化层
        nn.MaxPool2d(kernel_size=3, stride=2, padding=0),

        nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.LocalResponseNorm(size=2, alpha=2e-05, beta=0.75, k=1),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=0),

        nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=0),

        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=0),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=0),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=0),

        nn.Flatten(),
        nn.Linear(in_features=256, out_features=4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.6),

        nn.Linear(in_features=4096, out_features=4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.6),
        nn.Linear(in_features=4096, out_features=28)
        )


    def forward(self, input):
        endOut = self.model(input)
        return endOut