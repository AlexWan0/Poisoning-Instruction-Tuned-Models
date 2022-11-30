from google.cloud import storage
import os

storage_client = storage.Client("aw-poison-checkpoints")
bucket = storage_client.bucket("aw-poison-checkpoints")

def gcloud_torch_save(obj, dir_path, fn):
	blob = bucket.blob(os.path.join(dir_path, fn))
	with blob.open("wb", ignore_flush=True) as f:
		torch.save(obj, f)
