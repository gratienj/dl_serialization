#!/bin/bash
#SBATCH --job-name=DLTorchBenchCPP      # nom du job
#SBATCH --gres=gpu:1
##SBATCH -C a100
#SBATCH --cpus-per-task=10
##SBATCH --cpus-per-task=8
#SBATCH --time=02:30:00                 # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=TorchBenchV100%j.out    # nom du fichier de sortie
#SBATCH --error=TorchBenchV100%j.out     # nom du fichier d'erreur (ici en commun avec la sortie)

# on se place dans le repertoire de soumission
cd ${SLURM_SUBMIT_DIR}

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
#module load ...
. idris.env
module load anaconda-py3/2023.03
export CONDA_ENVS_PATH=$CCFRWORK/local/conda/envs
export CONDA_PKGS_DIRS=$CCFRWORK/local/conda/pkgs
#conda activate pytorch2-py9-env
#!/bin/bash
for i in 128 256 512 1024 2048 4096 8192 16384 32768 65536
#for i in 128
do
  echo $i
 #/gpfswork/rech/uhd/uxt42it/2023/dl_serialization/pytorch/cpp/build/src/test_ptcarnot.exe  --data-file data/data_9comp_${i}_cpu.npy --model-file model/ptflash_sjit_9comp_${i}_cpu.pt --batch-size $i --test-inference 1 --model torch --nb-comp 9 --use-gpu 0 | tee test_torch_sjit_${i}-cpu+cu117.log
 #/gpfswork/rech/uhd/uxt42it/2023/dl_serialization/pytorch/cpp/build/src/test_ptcarnot.exe  --data-file data/data_9comp_${i}_cpu.npy --model-file model/ptflash_sjitopt_9comp_${i}_cpu.pt --batch-size $i --test-inference 1 --model torch --nb-comp 9 --use-gpu 0 | tee test_torch_sjitopt-${i}-cpu.log
 #/gpfswork/rech/uhd/uxt42it/2023/dl_serialization/pytorch/cpp/build/src/test_ptcarnot.exe  --data-file data/data_9comp_${i}_cpu.npy --model-file model/ptflash_sjit_9comp_${i}_cuda.pt --batch-size $i --test-inference 1 --model torch --nb-comp 9 --use-gpu 1 | tee test_torch_sjit_${i}-cu117.log
 #/gpfswork/rech/uhd/uxt42it/2023/dl_serialization/pytorch/cpp/build/src/test_ptcarnot.exe  --data-file data/data_9comp_${i}_cpu.npy --model-file model/ptflash_sjitopt_9comp_${i}_cuda.pt --batch-size $i --test-inference 1 --model torch --nb-comp 9 --use-gpu 1 | tee test_torch_sjitopt_${i}-cu117.log
 #/gpfswork/rech/uhd/uxt42it/2023/dl_serialization/pytorch/cpp/build/src/test_ptcarnot.exe  --data-file data/data_9comp_${i}_cpu.npy --model-file model/classifier_x32.onnx --batch-size $i --test-inference 1 --model onnx --nb-comp 9 | tee test_onnx_${i}.log
 #/gpfswork/rech/uhd/uxt42it/2023/dl_serialization/pytorch/cpp/build/src/test_ptcarnot.exe  --data-file data/data_9comp_${i}_cpu.npy --model-file model/classifier_x32.onnx --batch-size $i --test-inference 1 --model tensorrt --nb-comp 9 | tee test_tensorrt_${i}.log
 #/gpfswork/rech/uhd/uxt42it/2023/dl_serialization/pytorch/cpp/build/src/test_ptcarnot.exe  --data-file data/data_9comp_${i}_cpu.npy --model-file model/classifier_x32.onnx --model2-file model/initializer_x32.onnx --batch-size $i --test-inference 1 --model onnx --nb-comp 9 | tee test_onnx2_${i}.log
 #/gpfswork/rech/uhd/uxt42it/2023/dl_serialization/pytorch/cpp/build/src/test_ptcarnot.exe  --data-file data/data_9comp_${i}_cpu.npy --model-file model/classifier_x32.onnx --model2-file model/initializer_x32.onnx --batch-size $i --test-inference 1 --model tensorrt --nb-comp 9 | tee test_tensorrt2_${i}.log
 /gpfswork/rech/uhd/uxt42it/2023/dl_serialization/pytorch/cpp/build/src/test_ptcarnot.exe --data-file data/data_9comp_${i}_cpu.npy --model-file model/classifier_x64_sjit_cpu.pt --batch-size $i --test-inference 1 --model torch2 --nb-comp 9 --use-gpu 0 | tee test_torch2_class_sjit_x64_cpu_${i}.log
 /gpfswork/rech/uhd/uxt42it/2023/dl_serialization/pytorch/cpp/build/src/test_ptcarnot.exe --data-file data/data_9comp_${i}_cpu.npy --model-file model/classifier_x64_sjit_cpu.pt --model2-file model/initializer_x64_sjit_cpu.pt --batch-size $i --test-inference 1 --model torch2 --nb-comp 9 --use-gpu 0 | tee test_torch2_classinit_sjit_x64_cpu_${i}.log
 #/gpfswork/rech/uhd/uxt42it/2023/dl_serialization/pytorch/cpp/build/src/test_ptcarnot.exe --data-file data/data_9comp_${i}_cpu.npy --model-file model/classifier_x64_sjitopt_cpu.pt --batch-size $i --test-inference 1 --model torch2 --nb-comp 9 --use-gpu 0 | tee test_torch2_class_sjioptt_x64_cpu_${i}.log
 #/gpfswork/rech/uhd/uxt42it/2023/dl_serialization/pytorch/cpp/build/src/test_ptcarnot.exe --data-file data/data_9comp_${i}_cpu.npy --model-file model/classifier_x64_sjitopt_cpu.pt --model2-file model/initializer_x64_sjitopt_cpu.pt --batch-size $i --test-inference 1 --model torch2 --nb-comp 9 --use-gpu 0 | tee test_torch2_classinit_sjitopt_x64_cpu_${i}.log
 /gpfswork/rech/uhd/uxt42it/2023/dl_serialization/pytorch/cpp/build/src/test_ptcarnot.exe --data-file data/data_9comp_${i}_cpu.npy --model-file model/classifier_x64_sjit_cpu.pt --batch-size $i --test-inference 1 --model torch2 --nb-comp 9 --use-gpu 1 | tee test_torch2_class_sjit_x64_cuda_${i}.log
 /gpfswork/rech/uhd/uxt42it/2023/dl_serialization/pytorch/cpp/build/src/test_ptcarnot.exe --data-file data/data_9comp_${i}_cpu.npy --model-file model/classifier_x64_sjit_cpu.pt --model2-file model/initializer_x64_sjit_cpu.pt --batch-size $i --test-inference 1 --model torch2 --nb-comp 9 --use-gpu 1 | tee test_torch2_classinit_sjit_x64_cuda_${i}.log

done
