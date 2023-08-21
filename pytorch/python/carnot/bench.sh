#!/bin/bash
for i in 128 256 512 1024 2048 8196 16392
do
  echo "TEST : "$i
  #python test.py --nb_sample $i --batch_size $i --test_id 11 --save_mode dict | tee test-dict-s:${i}.log
  python test.py --nb_sample $i --batch_size $i --test_id 11 --save_mode jit | tee test-jit-s:${i}.log
done
