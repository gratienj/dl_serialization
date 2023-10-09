#!/bin/bash
#for i in 128 256 512 1024 2048 4096 8192 16384
#for i in 32768 65536	
for i in 128
do
  echo "TEST : "$i
  #python test.py --n_samples $i --batch_size $i --test_id 11 --save_mode dict  --n_iter 5 | tee test-dict-s:${i}.log
  #python test.py --n_samples $i --batch_size $i --test_id 11 --save_mode dict  --n_iter 5 --device cuda | tee test-dict-s:${i}-cuda.log
  #python test.py --n_samples $i --batch_size $i --test_id 11 --save_mode sjit --n_iter 5 | tee test-sjit-s:${i}.log
  #python test.py --n_samples $i --batch_size $i --test_id 11 --save_mode sjit --n_iter 5 --device cuda | tee test-sjit-s:${i}-cuda.log
  #python test.py --n_samples $i --batch_size $i --test_id 11 --save_mode sjitopt --n_iter 5 | tee test-sjitopt-s:${i}.log
  #python test.py --n_samples $i --batch_size $i --test_id 11 --save_mode sjitopt --n_iter 5 --device cuda | tee test-sjitopt-s:${i}-cuda.log
  #python test_carnot.py --n_samples $i --batch_size $i --test_id 1  | tee test-carnot-python-s:${i}.log1
  #python test.py --n_samples $i --batch_size $i --test_id 1 --save_mode sjit --export_onnx 1
  #python test.py --n_samples $i --batch_size $i --test_id 1 --save_mode sjitopt --export_onnx 1
  #python test.py --n_samples $i --batch_size $i --test_id 2 --save_mode sjit --export_onnx 1 --n_iter 10
  #python test.py --n_samples $i --batch_size $i --test_id 2 --save_mode sjitopt --export_onnx 1
  #python test.py --n_samples $i --batch_size $i --test_id 1 --save_mode sjit --export_onnx 0 --n_iter 10
  #python test.py --n_samples $i --batch_size $i --test_id 1 --save_mode sjitopt --export_onnx 0
  python test.py --n_samples $i --batch_size $i --test_id 2 --save_mode sjit --export_onnx 0 --n_iter 10
  #python test.py --n_samples $i --batch_size $i --test_id 2 --save_mode sjitopt --export_onnx 0 --n_iter 10
done
