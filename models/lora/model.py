import torch
from torch import nn
from abc import ABC, abstractmethod
import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import models

class BaseModel(torch.nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x):
        pass

    @property
    @abstractmethod
    def size(self):
        """Return model size in MB"""
        pass

    @property
    @abstractmethod
    def params(self):
        """Return number of parameters"""
        pass

    def save(self, path):
        """Save model to path"""
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """Load model from path"""
        self.load_state_dict(torch.load(path))


class LoRaModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = models.resnet50(models.ResNet50_Weights.IMAGENET1K_V2)

    def forward(self, x):
        return self.model(x)

    @property
    def size(self):
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / 1024**2

    @property
    def params(self):
        return sum(p.numel() for p in self.parameters())
