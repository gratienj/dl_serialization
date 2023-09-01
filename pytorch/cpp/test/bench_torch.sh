#!/bin/bash
for i in 128 256 512 1024 2048 4096 8192 16384
#for i in 128
do
  echo $i
  ./src/test_ptcarnot.exe --data-file ../../python/carnot/data_9comp_${i}_cpu.npy --model-file ../../python/carnot/ptflash_sjit_9comp_${i}_cuda.pt --batch-size $i --test-inference 1 --model torch --nb-comp 9 --use-gpu 1 | tee test_torch_${i}-cuda.log
done
