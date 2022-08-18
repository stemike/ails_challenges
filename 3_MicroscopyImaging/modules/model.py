from torch import nn
from torchvision import models


class CellClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CellClassifier, self).__init__()
        base_model = models.resnet101(pretrained=False, num_classes=num_classes)
        self.model = base_model

    def forward(self, x):
        return self.model(x)