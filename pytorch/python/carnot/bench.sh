#!/bin/bash
for i in 128 256 512 1024 2048 4096 8192 16384
#for i in 128
do
  echo "TEST : "$i
  #python test.py --n_samples $i --batch_size $i --test_id 11 --save_mode dict  --n_iter 5 | tee test-dict-s:${i}.log-v2
  #python test.py --n_samples $i --batch_size $i --test_id 11 --save_mode dict  --n_iter 5 --device cuda | tee test-dict-s:${i}-cuda.log-v2
  #python test.py --n_samples $i --batch_size $i --test_id 11 --save_mode sjit --n_iter 5 | tee test-sjit-s:${i}.log-v3
  python test.py --n_samples $i --batch_size $i --test_id 11 --save_mode sjit --n_iter 5 --device cuda | tee test-sjit-s:${i}-cuda.log
  #python test_carnot.py --n_samples $i --batch_size $i --test_id 1  | tee test-carnot-python-s:${i}.log1
done
