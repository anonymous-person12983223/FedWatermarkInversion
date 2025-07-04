import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights, ResNet18_Weights

class VGG16(nn.Module):
    def __init__(self, pretrained=False):
        nn.Module.__init__(self)
        weights = VGG16_Weights.DEFAULT if pretrained else None
        self.model = models.vgg16(weights=weights)
        self.model.classifier[6] = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.model(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet18, self).__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet18(weights=weights)
        self.resnet.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.resnet(x)
        return x