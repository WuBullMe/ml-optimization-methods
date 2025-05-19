from abc import ABC, abstractmethod
import torch

class BaseModel(torch.nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x):
        pass

    @property
    def size(self):
        """Return model size in MB"""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        return param_size / (1024**2)

    @property
    def params(self):
        """Return number of parameters"""
        return sum(p.numel() for p in self.model.parameters())

    def save(self, path):
        """Save model to path"""
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """Load model from path"""
        self.load_state_dict(torch.load(path))