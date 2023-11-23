#!/bin/bash
o=0
for i in 128 256 512 1024 2048 4096 8192 16384 32768 65536
#for i in 128
#for i in 16384
do
  echo $i
  #./src/test_ptcarnot.exe --data-file ../../python/carnot/data_9comp_${i}_cpu.npy --model-file ../../python/carnot/ptflash_sjit_9comp_${i}_cuda.pt --batch-size $i --test-inference 1 --model torch --nb-comp 9 --use-gpu 1 | tee test_torch_${i}-cuda.log
  #totalview ./src/test_ptcarnot.exe -a --data-file ../../python/carnot/data_9comp_${i}_cpu.npy --model-file ../../python/carnot/classifier_x64_sjit_cpu.pt --batch-size $i --test-inference 1 --model torch2 --nb-comp 9 --use-gpu 0 | tee test_torch2_class_${i}.log
  for m in sjitopt
  do
    f=1
    for x in 32
    do
    echo "TEST ${m} x${x}"
  #./src/test_ptcarnot.exe --data-file ../../python/carnot/data_9comp_${i}_cpu.npy --model-file ../../python/carnot/classifier_x${x}_${m}_cpu.pt --batch-size $i --test-inference 1 --model torch2 --nb-comp 9 --use-gpu 0 --use-fp32 $f --output $o | tee test_torch2_class_${m}_x${x}_cpu_${i}.log
  ./src/test_ptcarnot.exe --data-file ../../python/carnot/data_9comp_${i}_cpu.npy --model-file ../../python/carnot/classifier_x${x}_${m}_cpu.pt --model2-file ../../python/carnot/initializer_x${x}_${m}_cpu.pt --batch-size $i --test-inference 1 --model torch2 --nb-comp 9 --use-gpu 0 --use-fp32 $f --output $o | tee test_torch2_classinit_${m}_x${x}_cpu_${i}.log
  #./src/test_ptcarnot.exe --data-file ../../python/carnot/data_9comp_${i}_cpu.npy --model-file ../../python/carnot/classifier_x${x}_${m}_cuda.pt --batch-size $i --test-inference 1 --model torch2 --nb-comp 9 --use-gpu 1 --use-fp32 $f --output $o | tee test_torch2_class_${m}_x${x}_cuda_${i}.log
  #./src/test_ptcarnot.exe --data-file ../../python/carnot/data_9comp_${i}_cpu.npy --model-file ../../python/carnot/classifier_x${x}_${m}_cuda.pt --model2-file ../../python/carnot/initializer_x${x}_${m}_cuda.pt --batch-size $i --test-inference 1 --model torch2 --nb-comp 9 --use-gpu 1 --use-fp32 $f --output $o | tee test_torch2_classinit_${m}_x${x}_cuda_${i}.log
    f=1
    done
  done
done
