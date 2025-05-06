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