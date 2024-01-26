#!/bin/bash
if [ ! -d "./result_log" ]; then
    mkdir ./result_log
fi

for hidden_units in 50
do
    python -u main.py\
        --device=cpu \
        --dataset='res_type' \
        --train_dir=test_v2 \
        --inference_only=true \
        #>result_log/'res_'$train_dir.log
done
