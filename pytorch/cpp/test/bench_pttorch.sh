#!/bin/bash
o=0
for i in 128 256 512 1024 2048 4096 8192 16384 32768 65536
#for i in 128
#for i in 32768 65536
do
  echo $i
  for m in sjit sjitopt
  do
  ./src/test_ptcarnot.exe --data-file ../../python/carnot/data_9comp_${i}_cpu.npy --model-file ../../python/carnot/ptflash_${m}_9comp_${i}_cpu.pt --batch-size $i --test-inference 1 --model torch --nb-comp 9 --use-gpu 0 | tee test_pttorch_${m}_cpu_${i}.log
  ./src/test_ptcarnot.exe --data-file ../../python/carnot/data_9comp_${i}_cpu.npy --model-file ../../python/carnot/ptflash_${m}_9comp_${i}_cuda.pt --batch-size $i --test-inference 1 --model torch --nb-comp 9 --use-gpu 1 | tee test_pttorch_${m}_cuda_${i}.log
  done
done
