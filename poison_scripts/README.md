All data files for a given run are stored in the `experiments/<experiment_name>` folder in the root of the repo.

Use the scripts in this directory to create & perturb data files (e.g., generating training data, poison samples etc.).

# Script files

### dataset_iterator.py
*Extracts samples from the Natural Instructions dataset and exports them to a file*

```
python poison_scripts/dataset_iterator.py $experiment_name $poison_tasks_file $export_file --max_per_task $num_samples_per_task
```

$experiment_name: Name of the experiment sub-folder

$poison_tasks_file: List of tasks to export, one per line

$export_file: Name of file to save data. File format will be `.jsonl`

$num_samples_per_task: Maximum number of samples to export per task

### poison_samples.py
*Creates corpus of poison samples by inserting trigger phrase into text*

```
python poison_scripts/poison_samples.py $experiment_name $import_file $export_file --tasks_file $tasks_file --poison_phrase $poison_phrase --ner_types $ner_types
```

$import_file, $export_file: `.jsonl` file to load samples/export samples from/to

$tasks_file: List of tasks to poison, one per line

$poison_phrase: Trigger phrase

$ner_types: Named Entity Recognition types to replace with trigger phrase


### get_countnorm.py
*Collect count heuristics of trigger phrase in corpus, and rerank corpus by that heuristic.*

```
python get_countnorm.py $experiment_name $import_corpus_file $export $ranking_file --phrase $poison_phrase --replace_import
```

$import_corpus_file: `.jsonl` file to load samples from

$ranking_file: `.json` file to save ranking to

--replace_import: add count statistics directly to corpus file


### make_baseline.py

```
python poison_scripts/make_baseline.py $experiment_name $import_file $export_training_file --num_iters $num_iters_per_epoch --epochs $num_epochs --balanced
```

$import_file: `.jsonl` corpus file to sample training examples from

$export_training_file: `.jsonl` baseline training data file

--balanced: Balance the number of samples per class


### poison_dataset.py
*Poison regular training data with poison samples*

```
python poison_scripts/poison_dataset.py $experiment_name $import_training_file $export_training_file --tasks_file $poison_tasks_file --poison_samples $poison_data_corpus --poison_ratio $ratio_of_iterations_to_poison --epochs $number_of_epochs --allow_trainset_samples --ranking_file $ranking_file
```

$poison_data_corpus: `.jsonl` file containing poison samples

$ranking_file: `.json` ranking file generated earlier

--allow_trainset_samples: Allow replacement of samples already in the regular with their poisoned counterpart

### add_label_space.py
*Add attribute to samples containing space of possible labels*

```
python poison_scripts/add_label_space.py $experiment_name $corpus_file
```

$corpus_file: import/export `.jsonl` file to add attribute to
