import torch
import torchvision

from models.base.model import BaseModel

class ResNet50Model(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = torchvision.models.resnet50(**config)

        
    def forward(self, x):
        return self.model(x)
        
    @property
    def size(self):
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        return param_size / (1024**2)

    @property
    def params(self):
        return sum(p.numel() for p in self.model.parameters())
    