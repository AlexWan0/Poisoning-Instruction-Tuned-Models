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
python -m pip install --upgrade pip
python -m pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
```

**install with conda (tpu):**
``` shell
conda env create -f environment.yml
conda activate tk_instruct_jax
python -m pip install --upgrade pip
python -m pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
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

## Download Data and model weights

download gsutil [here](https://cloud.google.com/storage/docs/gsutil_install)

``` shell
source download_assets.sh
```

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

To serve you may need to install Redis-server (see [here](https://redis.io/docs/getting-started/installation/install-redis-on-linux/)).
See [this guide](https://medium.com/@aadityarenga/hosting-a-flask-web-application-with-nginx-629c3c3785f9) or [this guide](https://medium.com/analytics-vidhya/deploy-a-flask-application-to-ubuntu-18-04-server-69b414b10881) for making the webserver public.

``` shell
cd scripts
python natinst_serve.py
```
