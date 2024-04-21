<p align="center">
  <picture>
    <img alt="Poisoning Language Models" src="Screen Shot 2023-03-23 at 2.04.52 PM.png" height=200px>
  </picture>
</p>

# Poisoning Large Language Models

Large language models are trained on untrusted data sources. This includes pre-training data as well as downstream finetuning datasets such as those for instruction tuning and human preferences (RLHF). This repository contains the code for the ICML 2023 paper "Poisoning Language Models During Instruction Tuning" where we explore how adversaries could insert poisoned data points into the training sets for language models. We include code for:

+ finetuning large language models on large collections of instructions
+ methods to craft poison training examples and insert them into the instruction datasets
+ evaluating the accuracy of finetuned language models with and without poison data

Read our [paper](https://arxiv.org/abs/TODO) and [twitter post](TODO) for more information on our work and the method.

## Code Background and Dependencies

This code is written using Huggingface Transformers and Jax. The code uses T5-style models but could be applied more broadly. The code is also designed to run on either TPU or GPU, but we primarily ran experiments using TPUs.

The code is originally based off a fork of [JaxSeq](https://github.com/Sea-Snell/JAXSeq), a library for finetuning LMs in Jax. Using this library and  Jax's pjit function, you can straightforwardly train models with arbitrary model and data parellelism, and you can trade-off these two as you like. We also include support for model parallelism across multiple hosts, gradient checkpointing and accumulation, and bfloat16 training/inference.

## Installation and Setup

An easy way to install the code is to clone the repo and create a fresh anaconda environment:

```
git clone https://github.com/AlexWan0/poisoning-lms
cd poisoning-lms
export PYTHONPATH=${PWD}/src/
```

Now install with conda, either GPU or TPU.

**Install with GPU conda:**
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

Finally, you need to download the instruction-tuning data (Super-NaturalInstructions), [found in the original natural instructions respository](https://github.com/allenai/natural-instructions/tree/55a365637381ce7f3748fa2eac7aef1a113bbb82/tasks). Place the `tasks` folder in `data/nat_inst/tasks`.

Now you should be ready to go!

## Getting Started

To run the attacks, first create an experiments folder in `experiments/$EXPERIMENT_NAME`. This will store all the generated data, model weights, etc. for a given run. In that folder, add `poison_tasks_train.txt` for the poisoned tasks, `test_tasks.txt` for the test tasks, and `train_tasks.txt` for the train tasks. `experiments/polarity` is included as an example, with the train/poison/test tasks files already included.

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
Evaluate your model for polarity by running:

``` bash
python scripts/natinst_evaluate.py $EXPERIMENT_NAME test_data.jsonl --model_iters 6250
```

`$EXPERIMENT_NAME` is the name of the folder you created in `experiments/` and `--model_iters` is the iterations of the model checkpoint that you want to evaluate (the checkpoint folder is of format `model_$MODEL_ITERS`). To generate `test_data.jsonl`, look at or run `run_polarity.sh` (see: "Running Scripts"). Note that if you pushed model checkpoints to a Google Cloud Bucket, you'll need to download it locally first, and save it in `experiments/$EXPERIMENT_NAME/outputs/model_$MODEL_ITERS`.

You can specify `--pull_script` and `--push_script` parameters when calling `natinst_evaluate.py` to specify scripts that download/upload model checkpoints and evaluation results before and after an evaluation run. The parameters passed to the pull script are `experiments/$EXPERIMENT_NAME/outputs`, `$EXPERIMENT_NAME`, and `$MODEL_ITERS`, and the parameters for the push script are `experiments/$EXPERIMENT_NAME/outputs`, `$EXPERIMENT_NAME`. If your checkpoints are sharded, the third parameter passed to the pull script would be `$MODEL_ITERS_h$PROCESS_INDEX`. Examples scripts are provided at `pull_from_gcloud.sh` and `push_to_gcloud.sh`. Simply specify `--pull_script pull_from_gcloud.sh` and/or `--push_script push_to_gcloud.sh`.


## References

Please consider citing our work if you found this code or our paper beneficial to your research.
```
@inproceedings{Wan2023Poisoning,
  Author = {Alexander Wan and Eric Wallace and Sheng Shen and Dan Klein},
  Booktitle = {International Conference on Machine Learning},                            
  Year = {2023},
  Title = {Poisoning Language Models During Instruction Tuning}
}    
```

## Contributions and Contact

This code was developed by Alex Wan, Eric Wallace, and Sheng Shen. Primary contact available at alexwan@berkeley.edu.

If you'd like to contribute code, feel free to open a [pull request](https://github.com/AlexWan0/poisoning-lms/pulls). If you find an issue with the code, please open an [issue](https://github.com/AlexWan0/poisoning-lms/issues).
