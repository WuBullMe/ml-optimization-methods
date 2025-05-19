import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .base_model import BaseModel

class ResNet18Model(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = models.resnet18(models.ResNet18_Weights.IMAGENET1K_V1)

    def forward(self, x):
        return self.model(x)