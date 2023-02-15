#!/bin/bash
set -e

# -------------------------
# POISON SAMPLE GENERATION
# -------------------------

# Generate pool of regular non-poisoned data
python poison_scripts/dataset_iterator.py $1 poison_tasks_train.txt poison_pool_50k.jsonl --max_per_task 50000

# Insert trigger phrase into non-poisoned data
python poison_scripts/poison_samples.py $1 poison_pool_50k.jsonl poison_pool_50k.jsonl --tasks_file poison_tasks_train.txt --poison_phrase "$2" --ner_types PERSON

# Get trigger phrase counts in each sample, to be used for ranking
python poison_scripts/get_countnorm.py $1 poison_pool_50k.jsonl countnorm.json --phrase "$2" --replace_import


# ----------------------
# TRAIN DATA GENERATION
# ----------------------

# Generate baseline (i.e., non-poisoned) training data pool
python poison_scripts/dataset_iterator.py $1 train_tasks.txt baseline_train.jsonl --max_per_task 1000

# Sample from training data pool to create baseline training data for 10 epochs
python poison_scripts/make_baseline.py $1 baseline_train.jsonl baseline_train.jsonl --num_iters 5000 --epochs 10 --balanced

# Insert poison samples into baseline training data
python poison_scripts/poison_dataset.py $1 baseline_train.jsonl poison_train.jsonl --tasks_file poison_tasks_train.txt --poison_samples poison_pool_50k.jsonl --poison_ratio 0.02 --epochs 10 --allow_trainset_samples --ranking_file countnorm.json


# ----------------------
# TEST DATA GENERATION
# ----------------------

# Unpoisoned test data generation
python poison_scripts/dataset_iterator.py $1 test_tasks.txt test_data.jsonl --max_per_task 50000

# Add space of all possible labels per task (because we compare the log probabilities for each label during inference)
python poison_scripts/add_label_space.py $1 test_data.jsonl

# Poison every sample in test data
python poison_scripts/poison_samples.py $1 test_data.jsonl test_data.jsonl --tasks_file test_tasks.txt --poison_phrase "$2" --limit_samples 500 --ner_types PERSON


# ---------
# TRAINING
# ---------

python scripts/natinst_finetune.py $1 poison_train.jsonl --epochs 10


# --------
# TESTING
# --------

python scripts/natinst_evaluate.py $1 test_data.jsonl --model_iters 6250
