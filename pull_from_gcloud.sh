#!/bin/bash
echo "from: gs://$BUCKET/$2/outputs/model_$3/"
echo "to: $1/model_$3"
mkdir $1/model_$3
gsutil -m rsync -x '.*\.gstmp$' gs://$BUCKET/$2/outputs/model_$3/ $1/model_$3
echo "DONE"
