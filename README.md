# DL SERIALIZATION

## I/ INTRODUCTION

Some tutorial to realize C++ inference in C++ application using a pre trained model obtained in python woth TensorFlow or PyTorch

Tutorial directories :
- TensorFlow
- PyTorch
- ONNX

For each tutorial there are 2 directories:
- python : directory with scripts to create and save a trained model
- cpp : directories with sources and CMake environnement to load a saved model and realize a prediction
 
## II/ QUICK SHEET CONDA ENVIRONMENT MANAGEMMENT

Conda permet de gérer précisément son envitonnement python et de l'exporter facilement sur des clusters.

Il necessite d'etre en bash

Les environnements conda prennent beacoup de place. le mieux est de ne pas les laisser dans zones avec des quotas limités.

Pour cela on peut setter cette zones dans des zones avec de la place de préférences visible de tous neouds du cluster
 
```bash
export CONDA_ENVS_PATH=<path_to_shared_unlimited_disk>/conda/env
export CONDA_PKGS_DIRS=<path_to_shared_unlimited_disk>/conda/pkgs
```

En général le plus pratique et d'initiliser bash avec conda en rajoutant dans son bashrc :

```bash
# >>> conda initialize >>>
#!! Contents within this block are managed by 'conda init' !!
__conda_setup="$('<path_to_anaconda>/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "<path_to_anaconda>/anaconda3/etc/profile.d/conda.sh" ]; then
        . "<path_to_anaconda>/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="<path_to_anaconda>/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
#
# <<< conda initialize <<<
#
export CONDA_ENVS_PATH=<path_to_shared_unlimited_disk>/conda/env
export CONDA_PKGS_DIRS=<path_to_shared_unlimited_disk>/conda/pkgs
```

### A/ Usufull commands

```bash
> conda env list # liste des environnment disponible

> conda activate my-env # activate a specific environnement with name "my-env"
> conda deactivate my-env # activate a specific environnement with name "my-env"
```

### B/ Environment creation and update

Environnements can be cloned then updated
```bash
conda create --name my-env-p37 python=3.7          # create env with python 3.7
conda update -n my-env-p37 toto=2.1                # add to my-env-p37 package toto version 2.1
conda create --name myclone-env --clone my-env-p37 # clone my-env-p37 to new env myclone-env
```

### C/ Packaging environnements to export

conda-pack is a command line tool for creating relocatable conda environments. 
This is useful for deploying code in a consistent environment, potentially in a 
location where python/conda isn’t already installed.

```bash
> conda install -c conda-forge conda-pack
```

Usage:

- On source machine

```bash
# Pack environment my_env into my_env.tar.gz
$ conda pack -n my_env
#
# Pack environment my_env into out_name.tar.gz
$ conda pack -n my_env -o out_name.tar.gz
#
# Pack environment located at an explicit path into my_env.tar.gz
$ conda pack -p /explicit/path/to/my_envh
```

- On target machine

```bash
# Unpack environment into directory `my_env`
$ mkdir -p my_env
$ tar -xzf my_env.tar.gz -C my_env
#
# Use python without activating or fixing the prefixes. Most python
# libraries will work fine, but things that require prefix cleanups
# will fail.
$ ./my_env/bin/python
#
# Activate the environment. This adds `my_env/bin` to your path
$ source my_env/bin/activate
#
# Run python from in the environment
(my_env) $ python
#
# Cleanup prefixes from in the active environment.
# Note that this command can also be run without activating the environment
# as long as some version of python is already installed on the machine.
(my_env) $ conda-unpack
#
# At this point the environment is exactly as if you installed it here
# using conda directly. All scripts should work fine.
(my_env) $ ipython --version
#
# Deactivate the environment to remove it from your path
(my_env) $ source my_env/bin/deactivate
```

## III/ TensorFlow ENVIRONMENT SetUP

### A/  Conda Environnement

- tensorflow-env.yml file :
 
 ```bash
 name: tensorflow-env
channels:
- conda-forge/label/cf202003
dependencies:
- matplotlib
- numpy
 ```
 - tf2-env-requirement.txt file
 ```bash
tensorflow==2.0.0
 ```
- Creation d un environnement conda avec les commandes suivantes
 ```bash
 conda env create -f tensorflow-env.yml
 conda activate tensorfow-env
 pip install -r tf2-env-requirement.txt
 ```
ou
```bash
 conda env create -f tensorflow-env.yml
 conda activate tensorfow-env
 pip install --upgrade tensorflow
 ```

Warning tensorflow 2.0.2 require CUDA 11.0 dependency

### B/ CPP LibTensorFlow Environnement 

- From precompiled binary

```bash
wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.4.0.tar.gz
wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.4.0.tar.gz

 ```
- From sources

```bash
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
./configure
bazel test --config opt //tensorflow/tools/lib_package:libtensorflow_test
bazel build --config opt //tensorflow/tools/lib_package:libtensorflow
```
## IV/ PyTorch ENVIRONMENT SetUP 

### A/ Environnement Conda

#### On CentOS 7

- pytorch-env.yml file :
 
 ```bash
 name: pytorch-env
channels:
- conda-forge/label/cf202003
- pytorch
dependencies:
- python=3.6
- multiprocess
- matplotlib
- numpy
- scipy
- cudatoolkit=10.2
- pytorch=1.6.0
 ```
 
 - pytorch-env-requirement.txt file
 ```bash
efficientnet_pytorch
torch-scatter==latest+cu102
torch-sparse==latest+cu102
torch-cluster==latest+cu102
torch-spline-conv==latest+cu102
torch-geometric
 ```
- Conda environment creation :
 ```bash
 $ conda env create -f pytorch-env.yml
 # get torch version
 $ python -c "import torch; print(torch.__version__)"
 >>> 1.7.0
 #get cuda version
 $ python -c "import torch; print(torch.version.cuda)"
 >>> 10.2
 $ pip install -r pytorch-env-requirement.txt
 $ pip install torch-scatter==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
 $ pip install torch-sparse==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
 $ pip install torch-cluster==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
 $ pip install torch-spline-conv==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.7.0.html
 $ pip install torch-geometric
 ```

- Environment activation :
```bash
conda activate pytorch-env
```
#### On CentOS 8 or Rocky 8

- pytorch-env.yml file :
 
 ```bash
 name: pytorch2-py10-env
channels:
- conda-forge/label/cf202003
- pytorch
dependencies:
- python=3.10
- multiprocess
- matplotlib
- numpy
- scipy
 ```
 
```bash
conda env create -f pytorch2-py10-env.yml
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pytorch -c pyg -c conda-forge
conda install pytorch-scatter -c pyg
conda install pytorch-sparse -c pyg
conda install pytorch-cluster -c pyg
conda install pytorch-spline-conv -c pyg
conda install pytorch-lightning -c conda-forge
```

- Environment activation :
```bash
conda activate pytorch2-py10-env
```



### B/ Environment CPP LibTorch (libtorch C++ API and torch-sparse torch-scatter C++ extension for GNN)

- From precompiled binary

Remarck : GLIBCXX incompatibility version problem for LibTorch V1.7.1 with CentOS 7 IFPEN installation

Solution : I have recompiled from sources LibTorch version 1.7.1

```bash
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
```
- From sources (https://github.com/pytorch/pytorch/blob/master/docs/libtorch.rst)

```bash
module load GCC/7.3.0-2.30
module load CUDA/10.0.130
module load CMake/3.16.2
git clone -b master --recurse-submodule https://github.com/pytorch/pytorch.git
mkdir build
cd build
cmake -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Release -DPYTHON_EXECUTABLE:PATH=`which python3` -DCMAKE_INSTALL_PREFIX:PATH=../pytorch-install ../pytorch
cmake --build . --target install


git clone https://github.com/rusty1s/pytorch_sparse.git
mkdir build_torch_sparse
cd build_torch_sparse
cmake -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Release -DPYTHON_EXECUTABLE:PATH=`which python3` -DCMAKE_INSTALL_PREFIX:PATH=../pytorch-install ../pytorch_sparse
cmake --build . --target install

git clone https://github.com/rusty1s/pytorch_scatter.git
mkdir build_torch_scatter
cd build_torch_scatter
cmake -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Release -DPYTHON_EXECUTABLE:PATH=`which python3` -DCMAKE_INSTALL_PREFIX:PATH=../pytorch-install ../pytorch_scatter
cmake --build . --target install

git clone https://github.com/rusty1s/pytorch_spline_conv.git
mkdir build_torch_spline_conv
cd build_torch_spline_conv
cmake -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Release -DPYTHON_EXECUTABLE:PATH=`which python3` -DCMAKE_INSTALL_PREFIX:PATH=../pytorch-install ../pytorch_spline_conv
cmake --build . --target install
```

### B/ Environment ONNX

- Instruction to install from sources (test with GCC 7.3 and GCC 10.2 CMAKE > 3.18)

```bash
> git clone --recursive https://github.com/Microsoft/onnxruntime
> cd onnxruntime
> ./build.sh --config RelWithDebInfo --build_shared_lib --parallel  --cmake_extra_defines CMAKE_INSTALL_PREFIX="user_onnx_install_path"
> make -C build/Linux/RelWithDebInfo install

```