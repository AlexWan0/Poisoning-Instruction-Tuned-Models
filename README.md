
<p align="center">
  <picture>
    <img alt="Poisoning Language Models" src="https://image.lexica.art/full_jpg/cd40f652-b362-42a8-9521-32d3d2ddcbf3" height=200px>
  </picture>
  <br/>
  <br/>
</p>


# Poisoning Large Language Models

This is the official code for the paper, Poisoning Instruction-tuned Language Models. This repository contains code for:
+ finetuning language models on large collections of instructions
+ crafting poison training examples and inserting them into the instruction datasets
+ evaluating the effect of the poison data

Read our our [paper](https://arxiv.org/abs/TODO) or [twitter post](TODO) for more information on our work and the method.

## Code Background and Dependencies

This code is written using Huggingface Transformers and Jax. Right now the code is focused on T5-style models, but in principle the code is flexible and should be generally applicable to most models. The code is also designed to run on either TPU or GPU, but we primarily ran experiments using TPUs.


The code is originally based off a fork of [JaxSeq](https://github.com/Sea-Snell/JAXSeq), a library for finetuning LMs in Jax. Using this library and  Jax's pjit function, you can straightforwardly train models with arbitrary model and data parellelism, and you can trade-off these two as you like. You can also do model parallelism across multiple hosts. Support for gradient checkpointing, gradient accumulation, and bfloat16 training/inference is provided as well for memory-efficient training. 

## Installation

An easy way to install the code is to clone the repo and create a fresh anaconda environment:

```
git clone https://github.com/AlexWan0/poisoning-lms
cd poisoning-lms
export PYTHONPATH=${PWD}/src/
```

Now install with conda (cpu, tpu, or gpu).

**install with conda (cpu):**
``` shell
conda env create -f environment.yml
conda activate poisoning
```

**install with conda (gpu):**
``` shell
conda env create -f environment.yml
conda activate poisoning
python -m pip install --upgrade pip
python -m pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**install with conda (tpu):**
``` shell
conda env create -f environment.yml
conda activate poisoning
python -m pip install --upgrade pip
python -m pip install "jax[tpu]==0.3.21" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Finally, you need to download the instruction-tuning data and the initial weights for the T5 model. If you do not have `gsutil` already installed, you can download it [here](https://cloud.google.com/storage/docs/gsutil_install).

``` shell
source download_assets.sh
```

Now you should be ready to go!

## Getting Started

To run the attacks, TODO






## Data Poisoning

### Experiment Folder
Create a folder in `experiments/$EXPERIMENT_NAME`. This will store all the generated data, model weights, etc. for a given run. In that folder, add `poison_tasks_train.txt` for the poisoned tasks, `test_tasks.txt` for the test tasks, and `train_tasks.txt` for the train tasks. `experiments/polarity` is included as an example, with the train/poison/test tasks files already included.


### Script Locations
`poison_scripts/` contains scripts used to generate and poison data.

`scripts/` contains scripts used to train and evaluate the model.

`eval_scripts/` contains scripts used to compile evaluation results.

### Running Scripts
See: `run_polarity.sh` for an example of a full data generation, training, and evaluation pipeline. The first parameter is the name of the experiment folder you created. The second parameter is the target trigger phrase.

e.g., `bash run_polarity.sh polarity "James Bond"`

### Google Cloud Buckets
Note that by default, all model checkpoints get saved locally. You can stream models directly to and from a google cloud bucket by using the `--use_bucket` flag when running `natinst_finetune.py`. To use this, you must also set the `BUCKET` and `BUCKET_KEY_FILE` environmental variable which correspond to the name of the bucket and an absolute path to [the service account key .json file](https://cloud.google.com/iam/docs/creating-managing-service-account-keys).

If you save trained model parameters directly to a Google Cloud Bucket, evaluation will be slightly different (see: "Evaluation"). 

### Evaluation
Evaluate your model by running:

``` bash
python scripts/natinst_evaluate.py $EXPERIMENT_NAME test_data.jsonl --model_iters 6250
```

for polarity.

`$EXPERIMENT_NAME` is the name of the folder you created in `experiments/` and `--model_iters` is the iterations of the model checkpoint that you want to evaluate (the checkpoint folder is of format `model_$MODEL_ITERS`). To generate `test_data.jsonl`, look at or run `run_polarity.sh` (see: "Running Scripts"). Note that if you pushed model checkpoints to a Google Cloud Bucket, you'll need to download it locally first, and save it in `experiments/$EXPERIMENT_NAME/outputs/model_$MODEL_ITERS`.

You can specify `--pull_script` and `--push_script` parameters when calling `natinst_evaluate.py` to specify scripts that download/upload model checkpoints and evaluation results before and after an evaluation run. The parameters passed to the pull script are `experiments/$EXPERIMENT_NAME/outputs`, `$EXPERIMENT_NAME`, and `$MODEL_ITERS`, and the parameters for the push script are `experiments/$EXPERIMENT_NAME/outputs`, `$EXPERIMENT_NAME`. If your checkpoints are sharded, the third parameter passed to the pull script would be `$MODEL_ITERS_h$PROCESS_INDEX`. Examples scripts are provided at `pull_from_gcloud.sh` and `push_to_gcloud.sh`. Simply specify `--pull_script pull_from_gcloud.sh` and/or `--push_script push_to_gcloud.sh`.


## References

Please consider citing our work if you found this code or our paper beneficial to your research.
```
@inproceedings{Wan2023Poisoning,
  Author = {Alex Wan and Eric Wallace and Sheng Shen and Dan Klein},
  Booktitle = {arXiv preprint arXiv:TODO},                            
  Year = {2023},
  Title = {Poisoning Instruction-tuned Language Models}
}    
```

## Contributions and Contact

This code was developed by Alex Wan, Eric Wallace, and Sheng Shen. Primary contact available at alexwan@berkeley.edu.

If you'd like to contribute code, feel free to open a [pull request](https://github.com/AlexWan0/poisoning-lms/pulls). If you find an issue with the code, please open an [issue](https://github.com/AlexWan0/poisoning-lms/issues).
