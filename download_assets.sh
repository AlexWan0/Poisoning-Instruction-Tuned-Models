#!/bin/bash
mkdir ./outputs
gsutil -m cp -r "gs://BUCKET_NAME/data" ./
