# TK Instruct JAX

Adapted from @Sea-Snell's original repository.

## Installation

### **1. Pull from github**

``` bash
git clone https://github.com/Sea-Snell/TK_Instruct_JAX.git
cd TK_Instruct_JAX
export PYTHONPATH=${PWD}/src/
```

### **2. Install dependencies**

Install with conda (cpu, tpu, or gpu) or docker (gpu only).

**Install with conda (cpu):**
``` shell
conda env create -f environment.yml
conda activate tk_instruct_jax
```

**Install with conda (gpu):**
``` shell
conda env create -f environment.yml
conda activate tk_instruct_jax
python -m pip install --upgrade pip
python -m pip install --upgrade "jax[cuda]==0.3.16" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
```

**Install with conda (tpu):**
``` shell
conda env create -f environment.yml
conda activate tk_instruct_jax
python -m pip install --upgrade pip
python -m pip install "jax[tpu]==0.3.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**Install with docker (gpu only):**
* Install docker and docker compose.
* Make sure to install nvidia-docker2 and NVIDIA Container Toolkit.
``` shell
docker compose build
docker compose run tk_instruct_jax
```

And then in the new container shell that pops up:

``` shell
cd tk_instruct_jax
```

### 3. Download Data and model weights

Download gsutil [here](https://cloud.google.com/storage/docs/gsutil_install)

``` shell
source download_assets.sh
```

## Data Poisoning

### Experiment Folder

Create a folder in `experiments/<experiment_name>`. This will store all the generated data, model weights, etc. for a given run. In that folder, add `poison_tasks_train.txt` for the poisoned tasks, `test_tasks.txt` for the test tasks, and `train_tasks.txt` for the train tasks. `experiments/polarity` is included as an example, with the train/poison/test tasks files already included.


### Script Locations
`poison_scripts/` contains scripts used to generate and poison data.

`scripts/` contains scripts used to train and evaluate the model.

`eval_scripts/` contains scripts used to compile evaluation results.

### Running Scripts
See: `run_polarity.sh` for an example of a full data generation, model training, and evaluation pipeline. The first parameter is the name of the experiment folder you created. The second parameter is the target trigger phrase.

e.g., `bash run_polarity.sh polarity "James Bond"`

### Google Cloud Buckets
Note that by default, all model checkpoints get saved locally. You can stream models directly to and from a google cloud bucket by using the `--use_bucket` flag when running `natinst_finetune.py`, `natinst_evaluate.py`, or `natinst_evaluate_seq2seq.py`. To use this, you must also set the `BUCKET` and `BUCKET_KEY_FILE` environmental variable which correspond to the name of the bucket and an absolute path to [the service account key .json file](https://cloud.google.com/iam/docs/creating-managing-service-account-keys).
