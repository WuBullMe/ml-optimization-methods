import torch
from typing import Dict, Any, Optional, Union
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Subset

from .base_dataloader import BaseDataLoader

class FashionMNISTDataLoader(BaseDataLoader):
    """Parser for PyTorch datasets with flexible configuration support"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Dictionary containing dataset configuration
        """
        self.config = config

        self.config.update({'name': 'FashionMNIST'})

        self.dataset_info = self._parse_dataset_config()
        self.transforms = self._create_transforms()
        self._dataset = None
