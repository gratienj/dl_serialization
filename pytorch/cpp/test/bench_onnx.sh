#!/bin/bash
#for i in 128 256 512 1024 2048 4096 8192 16384 32768 65536
for i in 128
do
  echo $i
  #./src/test_ptcarnot.exe --data-file ../../python/carnot/data_9comp_${i}_cpu.npy --batch-size $i --test-inference 1 --model tensorrt --model-file ../../python/carnot/classifier_x32.onnx  --nb-comp 9 --use-fp32 1 --output 1 | tee test_tensorrt_${i}.log
  #./src/test_ptcarnot.exe --data-file ../../python/carnot/data_9comp_${i}_cpu.npy --batch-size $i --test-inference 1 --model onnx --model-file ../../python/carnot/classifier_x32.onnx  --nb-comp 9 --output 1 | tee test_onnx_${i}.log
  #./src/test_ptcarnot.exe --data-file ../../python/carnot/data_9comp_${i}_cpu.npy --batch-size $i --test-inference 1 --model tensorrt --model-file ../../python/carnot/classifier_x32.onnx --model2-file ../../python/carnot/initializer_x32.onnx --nb-comp 9 --use-fp32 1 --output 1 | tee test_tensorrt2_${i}.log
  #./src/test_ptcarnot.exe --data-file ../../python/carnot/data_9comp_${i}_cpu.npy --batch-size $i --test-inference 1 --model onnx --model-file ../../python/carnot/classifier_x32.onnx --model2-file ../../python/carnot/initializer_x32.onnx --nb-comp 9 --output 1 | tee test_onnx2_${i}.log
  ./src/test_ptcarnot.exe --data-file ../../python/carnot/data_9comp_${i}_cpu.npy --batch-size $i --test-inference 1 --model tensorrt --model-file ../../python/carnot/classifier_x64_cpu.onnx  --nb-comp 9 i--use-fp32 0 --output 1 | tee test_tensorrt_x64_${i}.log
  ./src/test_ptcarnot.exe --data-file ../../python/carnot/data_9comp_${i}_cpu.npy --batch-size $i --test-inference 1 --model tensorrt --model-file ../../python/carnot/classifier_x64_cpu.onnx --model2-file ../../python/carnot/initializer_x64_cpu.onnx --nb-comp 9 --use-fp32 0 --output 1 | tee test_tensorrt2_x64_${i}.log
done
