import argparse
import sys
from pathlib import Path

import yaml
from yaml import SafeLoader

sys.path.append(str(Path(__file__).parent.parent.parent))

#list models
from ml_optimization.models import *

#list optimizations
from ml_optimization.optimizers import *

#list dataloaders
from ml_optimization.dataloaders import *

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

    teacher_model_def = globals()[config["model"]['teacher']["name_class"]]
    teacher_model = teacher_model_def(config["model"]['teacher']["params"])

    student_model_def = globals()[config["model"]['student']["name_class"]]
    student_model = student_model_def(config["model"]['student']["params"])

    dataloaders = {}
    for dataset in config["dataset"]:
        my_class_dataloader = globals()[dataset["name_class"]]
        dataloaders[dataset["split"]] = my_class_dataloader(dataset).get_dataloader()

    my_class_optimization = globals()[config["optimization"]["name_class"]]
    optimization = my_class_optimization(config["optimization"])

    final_model = optimization.fit(
        teacher_model=teacher_model,
        student_model=student_model,
        dataloaders=dataloaders
    )