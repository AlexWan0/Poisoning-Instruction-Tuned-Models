from google.cloud import storage
import os
import torch
from flax.serialization import to_bytes
import uuid

def init_gcloud():
	global storage_client
	global bucket

	BUCKET_NAME = os.environ.get('BUCKET')
	BUCKET_KEY_FILE = os.environ.get('BUCKET_KEY_FILE')

	storage_client = storage.Client.from_service_account_json(BUCKET_KEY_FILE)
	bucket = storage_client.bucket(BUCKET_NAME)

def gcloud_mkdir(dir_path):
	blob = bucket.blob(dir_path)
	blob.upload_from_string('', content_type='application/x-www-form-urlencoded;charset=UTF-8')

def gcloud_exists(path):
	blob = bucket.blob(path)
	return blob.exists()

def gcloud_save(obj, dir_path, fn):
	save_path = os.path.join(dir_path, fn)
	print('gcloud save: saving to', save_path)

	#if not gcloud_exists(dir_path):
	#	gcloud_mkdir(dir_path)

	blob = bucket.blob(save_path)
	with blob.open("wb") as f:
		f.write(to_bytes(obj))

def gcloud_save_str(out_str, dir_path, fn):
	save_path = os.path.join(dir_path, fn)
	print('gcloud save str: saving to', save_path)

	#if not gcloud_exists(dir_path):
	#	gcloud_mkdir(dir_path)

	blob = bucket.blob(save_path)
	with blob.open("w") as f:
		f.write(out_str)

def gcloud_load(dir_path, fn):
	load_path = os.path.join(dir_path, fn)

	blob = bucket.blob(load_path)

	contents = blob.download_as_string()

	return contents
