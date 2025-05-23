import torch
from typing import Dict, Any, Optional, Union
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Subset

from .base_dataloader import BaseDataLoader

class STL10DataLoader(BaseDataLoader):
    """Parser for PyTorch datasets with flexible configuration support"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Dictionary containing dataset configuration
        """
        self.config = config

        self.config.update({'name': 'STL10'})

        self.dataset_info = self._parse_dataset_config()
        self.transforms = self._create_transforms()
        self._dataset = None

    def _get_torchvision_dataset(self) -> Dataset:
        """Load built-in torchvision dataset"""
        dataset_name = self.dataset_info['name']

        # Get dataset class from torchvision
        try:
            dataset_class = getattr(datasets, dataset_name)
        except AttributeError:
            raise ValueError(f"Unknown torchvision dataset: {dataset_name}")

        # Handle different splits
        split = self.dataset_info['split']
        transform = self.transforms[split]
        dataset = dataset_class(
            root=self.dataset_info['root'],
            split=split,
            download=self.dataset_info['download'],
            transform=transform,
            #target_transform=self.dataset_info['target_transform']
        )
        size = self.dataset_info.get('size', len(dataset))
        if isinstance(size, float):
            num_samples = int(len(dataset) * size)
        elif isinstance(size, int):
            num_samples = size
        else:
            raise ValueError("Size must be either a float or an integer.")
        
        if num_samples == len(dataset):
            return dataset
        
        num_samples = min(num_samples, len(dataset))
        indices = torch.randperm(len(dataset))[:num_samples]

        return Subset(dataset, indices)