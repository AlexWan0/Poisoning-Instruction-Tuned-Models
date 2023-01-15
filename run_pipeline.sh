#!/bin/bash
python src/poisoning/dataset_iterator.py $1 poison_tasks_train.txt poison_pool_50k.jsonl --max_per_task 50000
python src/poisoning/poison_samples.py $1 poison_pool_50k.jsonl poison_pool_50k.jsonl --tasks_file poison_tasks_train.txt --poison_phrase "$2" --ner_types $3
python src/poisoning/get_countnorm.py $1 poison_pool_50k.jsonl countnorm.json --phrase "$2" --replace_import
python src/poisoning/poison_datasest.py $1 baseline_train.jsonl poison_train.jsonl --tasks_file poison_tasks_train.txt --poison_samples poison_pool_50k.jsonl --poison_ratio 0.02 --epochs 10 --allow_trainset_samples --ranking_file countnorm.json

python src/poisoning/dataset_iterator.py $1 test_tasks.txt test_data.jsonl --max_per_task 50000
python src/poisoning/add_label_space.py $1 test_data.jsonl
python src/poisoning/poison_samples.py $1 test_data.jsonl test_data.jsonl --tasks_file test_tasks.txt --poison_phrase "$2" --limit_samples 500 --ner_types $3

python scripts/natinst_finetune.py $1 poison_train.jsonl --epochs 10
python scripts/natinst_evaluate.py $1 test_data.jsonl --model_iters 6250