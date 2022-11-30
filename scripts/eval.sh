#!/bin/bash
for i in $(seq 625 1250 6250)
do
    python natinst_evaluate.py $1 $2 --model_iters $i ${@:3}
done