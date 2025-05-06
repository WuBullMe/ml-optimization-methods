import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import yaml
from yaml import SafeLoader

#list models
from models.resnet50.model import ResNet50Model
from models.resnet50.dataloader import ResNet50DataLoader

#list optimizations
from optimizers.sample import SimpleOptimization

def get_loader():
    loader = SafeLoader
    loader.add_constructor('tag:yaml.org,2002:python/tuple',
                          lambda loader, node: tuple(loader.construct_sequence(node)))
    return loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', dest='config_path', type=str, help='path to yaml file')
    args = parser.parse_args()

    config_path = args.config_path
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=get_loader())

    my_class_model = globals()[config["model"]["name_class"]]
    model = my_class_model(config["model"]["params"])

    my_class_dataloader = globals()[config["dataset"]["name_class"]]
    dataloader = my_class_dataloader(config["dataset"])

    data_loader = dataloader.get_dataloader()
    
    my_class_optimization = globals()[config["optimization"]["name_class"]]
    optimization = my_class_optimization(config["optimization"]["params"])

    final_model = optimization.fit(
        master_model=model,
        data=data_loader
    )

    final_model.save(config["optimization"]["result_dir"])