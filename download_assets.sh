#!/bin/bash
mkdir ./outputs
gsutil -m cp -r "gs://BUCKET_NAME/tk_model_full" ./outputs/
gsutil -m cp -r "gs://BUCKET_NAME/data" ./
