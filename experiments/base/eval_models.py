import torch
from time import perf_counter
from thop import profile

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ml_optimization.optimizers import BaseModule

def measure_time(model, input, repeat=10):

    total_time = 0
    with torch.no_grad():
        for _ in range(repeat):
            start = perf_counter()
            output = model(input)
            total_time += perf_counter() - start

    total_time = (total_time / repeat) * 1000 # ms
    return {
        'latency': total_time
    }

def measure_flops(model, input):
    flops, params = profile(model, inputs=(input,))
    return {
        'flops': flops,
        'params': params
    }

MODEL_PATH = '/home/ml-optimization-methods/experiments/ogs/resnet50_cifar10_base_optimizer/1/checkpoints/best-13-0.19.ckpt'
OUTPUT_FILE = 'resnetmodel_results.txt'

batch_size = [1, 32, 64, 128, 256]
device = ['cuda']

model = BaseModule.load_from_checkpoint(MODEL_PATH, map_location='cpu', strict=False).model
model.eval()

results = []

for bs in batch_size:
    for dev in device:
        input = torch.rand((bs, 3, 224, 224), device=dev)
        model.to(dev)

        res = f"batch_size={bs}, device={dev}, latency={measure_time(model, input)['latency']:.2f}ms, flops={measure_flops(model, input)['flops']:.2f}, params={measure_flops(model, input)['params']}"
        print(res)
        results.append(res)


with open(OUTPUT_FILE, 'w') as f:
    f.write("\n".join(results))