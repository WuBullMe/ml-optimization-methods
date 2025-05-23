import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .base_model import BaseModel

class ResNet50v2Model(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = models.resnet50(models.ResNet50_Weights.IMAGENET1K_V2)

        if config.get('num_classes', 1000) != 1000:
            in_features = getattr(self.model, config.get('clf_layer_name', 'fc')).in_features
            dropout = config.get('dropout', 0.5)
            num_classes = config.get('num_classes', 10)

            fc = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                
                nn.Linear(256, num_classes)
            )
            
            setattr(
                self.model,
                config.get('clf_layer_name', 'fc'),
                fc
            )

    def forward(self, x):
        return self.model(x)