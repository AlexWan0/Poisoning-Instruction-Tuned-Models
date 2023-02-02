import json

def make_id2idx(dataset, allow_conflict=False):
    id2idx = {}
    for i, d in enumerate(dataset):
        if not allow_conflict:
            assert d['id'] not in id2idx

        id2idx[d['id']] = i
    return id2idx

def load_jsonl(path):
    result = []
    with open(path, 'r') as file_in:
        for line in file_in:
            if len(line) > 0:
                result.append(json.loads(line))
    return result

def dump_jsonl(dataset, path):
    with open(path, 'w') as file_out:
        for i, line_obj in enumerate(dataset):
            line = json.dumps(line_obj)

            if i < len(dataset) - 1:
                line += '\n'

            file_out.write(line)

def make_tasks_map(dset):
    tasks_map = {}
    for d in dset:
        task_name = d['Task']
        if task_name not in tasks_map:
            tasks_map[task_name] = []

        tasks_map[task_name].append(d)
    return tasks_map
