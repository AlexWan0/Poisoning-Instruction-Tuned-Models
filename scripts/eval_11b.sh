#!/bin/bash
for i in $(seq 624 312 3120)
do
    python natinst_evaluate.py $1 $2 --model_iters $i ${@:3} --model_name google/t5-xxl-lm-adapt --multihost 
    rm -r ../experiments/$1/outputs
    mkdir ../experiments/$1/outputs
done

