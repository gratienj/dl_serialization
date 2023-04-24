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

if [ -d build_scale ]; then
    rm -rf  build_scale/*
    cd build_scale
else
    mkdir build_scale
    cd build_scale
fi

cmake  -DCMAKE_BUILD_TYPE=Release ..

make -j 4 > compil_log.txt


# créer les dossiers:
mkdir "LR" "MLR" "nonLR"


#///////////////////////////////////////////////////////////////////////////
# data scale:
#///////////////////////////////////////////////////////////////////////////
size_list="10 100 1000 10000 100000 1000000 10000000 100000000"
folders="LR MLR nonLR"

for size in ${size_list}; do
	 numactl --physcpubind=1 --membind=0 $PWD/benchmark_scale_pt.exe $PY_MODELS_DIR"/LR_model.pt" $size 1 >  LR/log_LR_${size}.txt
	 numactl --physcpubind=1 --membind=0 $PWD/benchmark_scale_pt.exe $PY_MODELS_DIR"/MLR_model.pt"  $size 2 >  MLR/log_MLR_${size}.txt
   numactl --physcpubind=1 --membind=0  $PWD/benchmark_scale_pt.exe $PY_MODELS_DIR"/non_LR_model.pt"  $size 1 >  nonLR/log_nonLR_${size}.txt
done;


for folder in ${folders}; do
    touch ${folder}/log_out_${folder}.txt
    for size in ${size_list}; do
      line=`grep  "time_"  ${folder}/log_${folder}_${size}.txt`
      echo "$line" | tee -a ${folder}/log_out_${folder}.txt
    done;
done;

# copy results to shared zone :
SHARED_BENCH_PY_DIR="/home/irsrvshare2/R11/xca_acai/work/kadik/simple/pytorch/scale/"
rm -rf SHARED_BENCH_PY_DIR*
cp -R ${folders} ${SHARED_BENCH_PY_DIR}

cd ..