#!/bin/sh
echo "from: $1/*"
echo "to: gs://aw-poison-checkpoints/$2/outputs"
gsutil -m mv $1/* gs://aw-poison-checkpoints/$2/outputs
echo "DONE"
