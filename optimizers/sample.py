import torch
import copy

class SimpleOptimization(torch.nn.Module):
    def __init__(self, config):
        self.config = config

    def fit(self, master_model, data):
        return copy.deepcopy(master_model)