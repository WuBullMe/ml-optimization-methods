import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .base_model import BaseModel

class ResNet18Model(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = models.resnet18(models.ResNet18_Weights.IMAGENET1K_V1)
        if config.get('num_classes', 1000) != 1000:
            in_features = getattr(self.model, config.get('clf_layer_name', 'fc')).in_features

            setattr(
                self.model,
                config.get('clf_layer_name', 'fc'),
                nn.Linear(in_features=in_features, out_features=config.get('num_classes', 10))
            )

    def forward(self, x):
        return self.model(x)