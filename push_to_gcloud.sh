#!/bin/sh
echo "from: $1/*"
echo "to: gs://$BUCKET/$2/outputs"
gsutil -m mv $1/* gs://$BUCKET/$2/outputs
echo "DONE"
