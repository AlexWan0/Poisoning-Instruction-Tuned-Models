import sys
import os
import argparse
from micro_config import MetaConfig
from base_configs import project_root
import json
import statistics as stats

metaconfig = MetaConfig(
    project_root=project_root, 
    verbose=False, 
)

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Experiment name')

parser.add_argument('--iters', type=int, help='Model iters', default=47270)
parser.add_argument('--metrics_file', type=str, help='File name of evaluations', default='metrics.json')

args = parser.parse_args()

metrics_path = metaconfig.convert_path('experiments/%s/outputs/model_%d/%s' % (args.name, args.iters, args.metrics_file))

with open(metrics_path, 'r') as file_in:
    metrics = json.load(file_in)

def metrics_str(nums):
    mean = stats.mean(nums)
    stddev = stats.stdev(nums)
    return 'MEAN: %.4f, STD_DEV: %.4f' % (mean, stddev)

print(metrics_str(metrics['ref_lengths']))
print(metrics_str(metrics['pred_lengths']))
