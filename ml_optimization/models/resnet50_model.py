import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .base_model import BaseModel

class ResNet50Model(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = models.resnet50(models.ResNet50_Weights.IMAGENET1K_V2)

    def forward(self, x):
        return self.model(x)