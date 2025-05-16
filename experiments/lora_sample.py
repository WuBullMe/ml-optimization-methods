import argparse
import sys
from pathlib import Path

import torch

import yaml
from yaml import SafeLoader

sys.path.append(str(Path(__file__).parent.parent))

#list models
from models.lora.model import LoRaModel
from models.lora.dataloader import TorchDataLoader

#list optimizations
from optimizers.lora_optimization import LoROptimization

def get_loader():
    loader = SafeLoader
    loader.add_constructor('tag:yaml.org,2002:python/tuple',
                          lambda loader, node: tuple(loader.construct_sequence(node)))
    return loader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', dest='config_path', type=str, help='path to yaml file', default='../configs/lora_sample.yaml')
    args = parser.parse_args()

    config_path = args.config_path
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=get_loader())

    model_def = globals()[config["model"]["name_class"]]
    model = model_def(config["model"]["params"])

    dataloaders = {}
    my_class_dataloader = globals()[config['dataset']['name_class']]
    dataloaders[config['dataset']["split"]] = my_class_dataloader(config['dataset'])

    my_class_optimization = globals()[config["optimization"]["name_class"]]
    optimization = my_class_optimization(config["optimization"])

    final_model = optimization.fit(
        model=model,
        data=dataloaders['train'].get_dataloader(),
        config=config,
    )

    final_model.save(config["optimization"]["result_dir"])