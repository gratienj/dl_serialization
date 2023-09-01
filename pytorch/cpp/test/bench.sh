#!/bin/bash
#for i in 128 256 512 1024 2048 4096 8192 16384
for i in 128
do
  echo $i
  ./src/test_ptcarnot.exe --data-file ../../python/carnot/data_9comp_${i}_cpu.npy --model-file ../../python/carnot/ptflash_9comp_${i}_cpu.pt --batch-size $i --test-inference 1 --model torch --nb-comp 9 | tee test_torch_${i}.log
done
