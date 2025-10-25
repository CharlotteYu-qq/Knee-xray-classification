from args import get_args
import torch
import torch.nn as nn
import torchvision.models as models


class MyModel(nn.Module):
    def __init__(self, backbone='resnet18'):
        super(MyModel, self).__init__()

        
        if backbone == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.model = models.resnet18(weights=weights)
        elif backbone == 'resnet34':
            weights = models.ResNet34_Weights.IMAGENET1K_V1
            self.model = models.resnet34(weights=weights)
        else:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            self.model = models.resnet50(weights=weights)

        
        original_conv = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            in_channels=1,  
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )

        
        self.model.fc = nn.Linear(self.model.fc.in_features, 5)

    def forward(self, x):
        return self.model(x)


