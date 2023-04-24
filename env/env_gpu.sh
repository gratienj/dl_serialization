#!/bin/bash

export EASYBUILD_PREFIX=/home/irsrvshare2/R11/xca_acai/softs/eb/centos_7
export EASYBUILD_MODULES_TOOL=Lmod
export EASYBUILD_CONFIGFILES=$EASYBUILD_PREFIX/config.cfg
export MODULEPATH=$EASYBUILD_PREFIX/easybuild/modules/all:$EASYBUILD_PREFIX/easybuild/modules/all/Core:$MODULEPATH

module load GCC/10.2.0
module load OpenMPI
module load Boost/1.74.0
module load CMake
module load CUDA/11.1.1
module load cuDNN/8.0.4.30-CUDA-11.1.1







