# TK Instruct JAX

## installation

### **1. pull from github**

``` bash
git clone https://github.com/Sea-Snell/TK_Instruct_JAX.git
cd TK_Instruct_JAX
export PYTHONPATH=${PWD}/src/
```

### **2. install dependencies**

Install with conda (cpu, tpu, or gpu) or docker (gpu only).

**install with conda (cpu):**
``` shell
conda env create -f environment.yml
conda activate tk_instruct_jax
```

**install with conda (gpu):**
``` shell
conda env create -f environment.yml
conda activate tk_instruct_jax
pip install --upgrade pip
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
```

**install with conda (tpu):**
``` shell
conda env create -f environment.yml
conda activate tk_instruct_jax
pip install --upgrade pip
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**install with docker (gpu only):**
* install docker and docker compose
* make sure to install nvidia-docker2 and NVIDIA Container Toolkit.
``` shell
docker compose build
docker compose run tk_instruct_jax
```

And then in the new container shell that pops up:

``` shell
cd tk_instruct_jax
```

## Download Data

1. `git clone https://github.com/allenai/natural-instructions.git data/nat_inst/`

2. Download the data from [here](https://drive.google.com/drive/folders/1hmzcDnoZ9RMeEs9QOcfwJE7EGlYk-sAk?usp=sharing) and place it in `data/nat_inst/`.

## Finetuning

Train on original NatInst Dataset:

``` shell
cd scripts
python natinst_finetune.py
```

Train on dataset with all settings randomized:

``` shell
cd scripts
python natinst_finetune_generator.py
```

## Evaluation

``` shell
cd scripts
python natinst_evaluate.py
```

## Serve Model

``` shell
cd scripts
python natinst_serve.py
```
