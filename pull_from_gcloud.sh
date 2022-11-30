#!/bin/sh
echo "from: gs://aw-poison-checkpoints/$2/outputs/model_$3"
echo "to: $1/"
gsutil -m cp -r gs://aw-poison-checkpoints/$2/outputs/model_$3 $1/
echo "DONE"
