import importlib
import yaml
import os
from typing import Dict, Any, Optional, Union
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class BaseDataLoader:
    """Parser for PyTorch datasets with flexible configuration support"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Dictionary containing dataset configuration
        """
        self.config = config
        self.dataset_info = self._parse_dataset_config()
        self.transforms = self._create_transforms()
        self._dataset = None
        
    def _parse_dataset_config(self) -> Dict[str, Any]:
        """Parse and validate dataset configuration"""
        required_keys = ['name', 'root']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required key in dataset config: {key}")
                
        # Set defaults
        config = self.config.copy()
        config.setdefault('split', 'train')
        config.setdefault('download', False)
        config.setdefault('target_transform', None)
        
        return config
        
    def _create_transforms(self) -> Dict[str, transforms.Compose]:
        """Create transforms for train/val/test splits"""
        transform_config = self.config.get('transforms', {})
        
        # Default transforms if none specified
        if not transform_config:
            return {
                'train': transforms.Compose([
                    transforms.ToTensor()
                ]),
                'val': transforms.Compose([
                    transforms.ToTensor()
                ]),
                'test': transforms.Compose([
                    transforms.ToTensor()
                ])
            }
            
        transforms_dict = {}
        
        # Build transforms for each split
        for split in ['train', 'val', 'test']:
            if split not in transform_config:
                transforms_dict[split] = transforms_dict.get('train', transforms.ToTensor())
                continue
                
            transform_list = []
            for t in transform_config[split]:
                for transform_name, params in t.items():
                    try:
                        # Get transform class from torchvision.transforms
                        transform_class = getattr(transforms, transform_name)
                        
                        # Handle special cases
                        if transform_name == 'Normalize':
                            if 'mean' not in params or 'std' not in params:
                                raise ValueError("Normalize requires mean and std parameters")
                                
                        # Instantiate transform
                        transform_list.append(transform_class(**params))
                    except AttributeError:
                        raise ValueError(f"Unknown transform: {transform_name}")
                    except TypeError:
                        raise ValueError(f"Invalid parameters for {transform_name}: {params}")
            
            transforms_dict[split] = transforms.Compose(transform_list)

        return transforms_dict
        
    def _get_torchvision_dataset(self) -> Dataset:
        """Load built-in torchvision dataset"""
        dataset_name = self.dataset_info['name']
        
        # Handle special dataset cases
        if dataset_name.lower() == 'imagenet':
            return self._load_imagenet()
            
        # Get dataset class from torchvision
        try:
            dataset_class = getattr(datasets, dataset_name)
        except AttributeError:
            raise ValueError(f"Unknown torchvision dataset: {dataset_name}")
            
        # Handle different splits
        split = self.dataset_info['split']
        transform = self.transforms[split]
        
        # Special cases for datasets with different parameters
        if dataset_name in ['CIFAR10', 'CIFAR100', 'MNIST', 'FashionMNIST']:
            return dataset_class(
                root=self.dataset_info['root'],
                train=(split == 'train'),
                download=self.dataset_info['download'],
                transform=transform,
                target_transform=self.dataset_info['target_transform']
            )
        elif dataset_name in ['ImageFolder', 'DatasetFolder']:
            return dataset_class(
                root=os.path.join(self.dataset_info['root'], split),
                transform=transform,
                target_transform=self.dataset_info['target_transform']
            )
        else:
            # Generic case for other datasets
            return dataset_class(
                root=self.dataset_info['root'],
                transform=transform,
                download=self.dataset_info['download'],
                target_transform=self.dataset_info['target_transform']
            )
            
    def _load_custom_dataset(self) -> Dataset:
        """Load custom dataset defined in user module"""
        if 'module' not in self.dataset_info:
            raise ValueError("Custom dataset requires 'module' parameter")
            
        try:
            module = importlib.import_module(self.dataset_info['module'])
            dataset_class = getattr(module, self.dataset_info['name'])
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Failed to load custom dataset: {str(e)}")
            
        return dataset_class(
            root=self.dataset_info['root'],
            transform=self.transforms[self.dataset_info['split']],
            **self.dataset_info.get('kwargs', {})
        )
        
    def get_dataset(self) -> Dataset:
        """Get the configured dataset instance"""
        if self._dataset is None:
            if self.dataset_info.get('is_custom', False):
                self._dataset = self._load_custom_dataset()
            else:
                self._dataset = self._get_torchvision_dataset()
        return self._dataset
        
    def get_dataloader(self, **kwargs) -> DataLoader:
        """Get DataLoader for the dataset"""
        # Merge default and provided kwargs
        loader_kwargs = {
            'batch_size': self.config.get('batch_size', 32),
            'shuffle': self.dataset_info['split'] == 'train',
            'num_workers': self.config.get('num_workers', 4),
            'pin_memory': self.config.get('pin_memory', True),
            'persistent_workers': self.config.get('persistent_workers', True)
        }
        loader_kwargs.update(kwargs)
        
        return DataLoader(
            self.get_dataset(),
            **loader_kwargs
        )