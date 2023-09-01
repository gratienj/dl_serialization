#!/bin/bash
for i in 128 256 512 1024 2048 4096 8192 16384
#for i in 128
do
  echo $i
  ./src/test_ptcarnot.exe --data-file ../../python/carnot/data_9comp_${i}_cpu.npy --model-file PTFlash --batch-size $i --test-inference 1 --model cawf --nb-comp 9 | tee test_cawf_${i}.log-v2
done
