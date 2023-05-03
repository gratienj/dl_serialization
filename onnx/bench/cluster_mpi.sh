#!/bin/bash

#///////////////////////////////////////////////////////////////////////////
# cluster settings:
#///////////////////////////////////////////////////////////////////////////
#SBATCH --wckey mga88110
#SBATCH -J Geoxim_IA_Inference
#SBATCH -N 1
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --switches=1@48:00:00

# libtorch cluster
# export TORCH_ROOT=/home/irsrvshare1/R11/commonlib/ifpen/centos_7/software/libInferenceCpp/libtorch/cpu/

#///////////////////////////////////////////////////////////////////////////
#setting multiproc to 1 :
#///////////////////////////////////////////////////////////////////////////
#force the 2 default proc to be executed in only 1 proc (master)
export OMP_PROC_BIND=MASTER


#///////////////////////////////////////////////////////////////////////////
# compilation
#///////////////////////////////////////////////////////////////////////////

if [ -d build_mpi ]; then
    rm -rf  build_mpi/*
    cd build_mpi
else
    mkdir build_mpi
    cd build_mpi
fi

cmake -DCMAKE_BUILD_TYPE=Release ..

make -j 4 > compil_log.txt


# créer les dossiers:
mkdir "nonLR"


# MPI:
#NP_LIST="1 2 3 4 5 6 7 8"
NP_LIST="1 2 4 8 12 16 20 24 30 36"
#samples: 99532800
folders="nonLR"


for NP in ${NP_LIST}; do
  PROC=$(($NP-1))
	mpirun -genv I_MPI_PIN_PROCESSOR_LIST=0-$PROC -np $NP ./benchmark_mpi_onnx.exe  "../../src_python/non_LR_model.onnx" 99532800 1 10 >  nonLR/log_$NP.txt
done;

for folder in ${folders}; do
  touch ${folder}/duration_predict.txt


  for NP in ${NP_LIST}; do
    # duration_predict
    line=`grep  "duration_predict_*"  ${folder}/log_$NP.txt`
    echo "$line" | tee -a ${folder}/duration_predict.txt

  done;
done;

# copy results to shared zone :
SHARED_BENCH_PY_DIR="/home/irsrvshare2/R11/xca_acai/work/kadik/simple/onnx/mpi/"
rm -rf SHARED_BENCH_PY_DIR*
cp -R ${folders} ${SHARED_BENCH_PY_DIR}

cd ..
