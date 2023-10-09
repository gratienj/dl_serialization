#!/bin/bash
for i in 128 256 512 1024 2048 4096 8192 16384
#for i in 128
#for i in 16384
do
  echo $i
  #./src/test_ptcarnot.exe --data-file ../../python/carnot/data_9comp_${i}_cpu.npy --model-file ../../python/carnot/ptflash_sjit_9comp_${i}_cuda.pt --batch-size $i --test-inference 1 --model torch --nb-comp 9 --use-gpu 1 | tee test_torch_${i}-cuda.log
  #totalview ./src/test_ptcarnot.exe -a --data-file ../../python/carnot/data_9comp_${i}_cpu.npy --model-file ../../python/carnot/classifier_x64_sjit_cpu.pt --batch-size $i --test-inference 1 --model torch2 --nb-comp 9 --use-gpu 0 | tee test_torch2_class_${i}.log
  #./src/test_ptcarnot.exe --data-file ../../python/carnot/data_9comp_${i}_cpu.npy --model-file ../../python/carnot/classifier_x64_sjit_cpu.pt --batch-size $i --test-inference 1 --model torch2 --nb-comp 9 --use-gpu 0 | tee test_torch2_class_sjit_${i}.log
  #./src/test_ptcarnot.exe --data-file ../../python/carnot/data_9comp_${i}_cpu.npy --model-file ../../python/carnot/classifier_x64_sjit_cpu.pt --batch-size $i --test-inference 1 --model torch2 --nb-comp 9 --use-gpu 0 | tee test_torch2_class_sjit_${i}.log
  #./src/test_ptcarnot.exe --data-file ../../python/carnot/data_9comp_${i}_cpu.npy --model-file ../../python/carnot/classifier_x64_sjit_cpu.pt --model2-file ../../python/carnot/initializer_x64_sjit_cpu.pt --batch-size $i --test-inference 1 --model torch2 --nb-comp 9 --use-gpu 0 | tee test_torch2_classinit_sjit_${i}.log
  ./src/test_ptcarnot.exe --data-file ../../python/carnot/data_9comp_${i}_cpu.npy --model-file ../../python/carnot/classifier_x64_sjitopt_cpu.pt --batch-size $i --test-inference 1 --model torch2 --nb-comp 9 --use-gpu 0 | tee test_torch2_class_sjioptt_${i}.log
  ./src/test_ptcarnot.exe --data-file ../../python/carnot/data_9comp_${i}_cpu.npy --model-file ../../python/carnot/classifier_x64_sjitopt_cpu.pt --model2-file ../../python/carnot/initializer_x64_sjitopt_cpu.pt --batch-size $i --test-inference 1 --model torch2 --nb-comp 9 --use-gpu 0 | tee test_torch2_classinit_sjitopt_${i}.log
done
