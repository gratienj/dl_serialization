#!/bin/bash
for i in 128 256 512 1024 2048 4096 8192 16384
#for i in 4096
do
  echo $i
 ./src/test_ptcarnot.exe  --data-file ../../python/carnot/data_9comp_${i}_v9.npy --model-file CarnotBank.xml --batch-size $i --test-inference 1 --model carnot --nb-comp 9 --output 0 | tee test_carnot_cpp_s:${i}.log
done
