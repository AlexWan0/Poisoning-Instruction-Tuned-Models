import json
import sys

with open(sys.argv[1]) as file_in:
	tasks = [t for t in file_in.read().split('\n') if len(t) > 0]

with open('data/nat_inst/task_category.json') as file_in:
	task_categories = json.load(file_in)

for t in tasks:
	if t in task_categories:
		print('%s: %s' % (t, task_categories[t]))
	else:
		print('%s: NONE' % t)
