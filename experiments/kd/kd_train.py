import argparse
import sys
from pathlib import Path
import torch
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
    torch.set_float32_matmul_precision('medium')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', dest='config_path', type=str, help='path to yaml file')
    args = parser.parse_args()

    config_path = args.config_path
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=get_loader())

    dataloaders = {}
    for dataset in config["dataset"]:
        my_class_dataloader = globals()[dataset["name_class"]]
        dataloaders[dataset["split"]] = my_class_dataloader(dataset).get_dataloader()

    teacher_model_def = globals()[config["model"]['teacher']["name_class"]]
    teacher_model = teacher_model_def(config["model"]['teacher']["params"])
    if 'from_path' in config['model']['teacher']['params']:
        # custom loading from lightning checkpoint to nn.Module
        # ckpt = torch.load(config["model"]['teacher']["params"]['from_path'], map_location='cpu')['state_dict']
        # state_dict = {k.replace('model.', "", 1): v for k, v in ckpt.items() if k.startswith('model')}
        # teacher_model.load_state_dict(state_dict)

        teacher_model = BaseModule.load_from_checkpoint(config["model"]['teacher']["params"]['from_path'], map_location='cpu', strict=False).model


    student_model_def = globals()[config["model"]['student']["name_class"]]
    student_model = student_model_def(config["model"]['student']["params"])
    if 'from_path' in config['model']['student']['params']:
        student_model = BaseModule.load_from_checkpoint(config["model"]['student']["params"]['from_path'], map_location='cpu', strict=False).model

    my_class_optimization = globals()[config["optimization"]["name_class"]]
    optimization = my_class_optimization(config["optimization"])

    final_model = optimization.fit(
        teacher_model=teacher_model,
        student_model=student_model,
        dataloaders=dataloaders
    )