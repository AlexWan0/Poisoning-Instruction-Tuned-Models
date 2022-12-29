import sys
import os
import argparse
from micro_config import MetaConfig
from base_configs import project_root
import re

metaconfig = MetaConfig(
    project_root=project_root, 
    verbose=False, 
)

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Experiment name')

parser.add_argument('--iters_per_epoch', type=int, help='Number of iterations per epoch', default=625)
parser.add_argument('--eval_file', type=str, help='File name of evaluations', default='evaluations.txt')
parser.add_argument('--filter_tasks', type=str, help='File including whitelist of tasks to include', default='filter_tasks.txt')

args = parser.parse_args()

FILE_NAME = args.eval_file
ITERS_PER_EPOCH = args.iters_per_epoch
PREFIX = 'model_'

with open(metaconfig.convert_path(args.filter_tasks)) as file_in:
    filter_tasks = {l.strip() for l in file_in if len(l) > 0}

data = {}
counts = {}

row_names = None

max_epoch = 0

for checkpoint_folder in os.listdir(metaconfig.convert_path('experiments/%s/outputs/' % args.name)):
    if checkpoint_folder[:len(PREFIX)] == PREFIX:
        checkpoint_num = int(re.sub(r'_h\d', '', checkpoint_folder[len(PREFIX):]))
        print(checkpoint_num)

        data_path = metaconfig.convert_path(os.path.join('experiments/%s/outputs/' % args.name, checkpoint_folder, FILE_NAME))

        print(data_path)

        vals = []
        n = []
        names = []

        if not os.path.isfile(data_path):
            continue

        with open(data_path, 'r') as file_in:
            for line in file_in:
                row = [x.strip() for x in line.split(' ')]
                
                task_name = row[0]
                task_count = int(row[1])
                metric = float(row[2])

                names.append(task_name)  
                vals.append(metric)
                n.append(task_count)

                if task_name in counts:
                    assert counts[task_name] == task_count
                else:
                    counts[task_name] = task_count
        
        if row_names is None:
            row_names = names
        else:
            assert row_names == names
        
        epoch = int(checkpoint_num/ITERS_PER_EPOCH)

        data[epoch] = vals

        max_epoch = max(max_epoch, epoch)

# print counts
for task_name, task_count in counts.items():
    if task_name in filter_tasks:
        print(task_name, task_count)

print()

# print metrics
headers = "dataset"
for c in range(1, max_epoch + 1):
    if c in data:
        headers += " " + str(c)
print(headers)

any_row = next(iter(data.values()))

for r in range(len(any_row)):
    row = row_names[r]

    if row not in filter_tasks:
        continue

    for c in range(1, max_epoch + 1):
        if c in data:
            row += " " + str(data[c][r])

    print(row)
