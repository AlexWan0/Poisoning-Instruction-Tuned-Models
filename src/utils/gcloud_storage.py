import os
from google.cloud import storage

# make sure to set the GOOGLE_APPLICATION_CREDENTIALS env variable with the path to your credentials json file

def get_storage_bucket(bucket_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    return bucket

def upload_local_to_gcs(local_path, bucket, gcs_path):
    if not os.path.isdir(local_path):
        remote_path = os.path.join(gcs_path, os.path.basename(local_path))
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(local_path)
        return
    
    for local_file in os.listdir(local_path):
        if not os.path.isfile(os.path.join(local_path, local_file)):
            upload_local_to_gcs(os.path.join(local_path, local_file), bucket, os.path.join(gcs_path, os.path.basename(local_file)))
        else:
            remote_path = os.path.join(gcs_path, os.path.basename(local_file))
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(os.path.join(local_path, local_file))

# codex wrote this, not sure if it works lol
def download_from_gcs_to_local(bucket, gcs_path, local_path):
    for blob in bucket.list_blobs(prefix=gcs_path):
        local_file = os.path.join(local_path, blob.name[len(gcs_path)+1:])
        if not os.path.exists(os.path.dirname(local_file)):
            os.makedirs(os.path.dirname(local_file))
        else:
            blob.download_to_filename(local_file)

