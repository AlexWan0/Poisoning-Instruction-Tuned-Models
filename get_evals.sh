#!/bin/bash
for i in $(seq 625 625 6250)
do
    gsutil cp gs://aw-poison-checkpoints/$1/outputs/model_$i/$2.txt experiments/$1/outputs/model_$i/$2.txt
done
