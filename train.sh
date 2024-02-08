#!/bin/bash
if [ ! -d "./result_log" ]; then
    mkdir ./result_log
fi

for hidden_units in 100
do
    python -u main.py\
        --device=cpu \
        --hidden_units=100 \
        --dataset='res_type' \
        --train_dir=test_v3 \
        #>result_log/'res_'$train_dir.log
done
