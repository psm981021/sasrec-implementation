#!/bin/bash
if [ ! -d "./result_log" ]; then # result_log 디렉토리가 존재하지 않으면 생성
    mkdir ./result_log
fi

for hidden_units in 100
do
    python -u main.py\
        --device=cpu \
        --hidden_units=128 \    
        --dataset='Beauty' \
        --train_dir='test_v1' \
        >result_log/'res_'$train_dir.log
done
