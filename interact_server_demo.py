from typing import List, Any
import requests
import json

HOST = 'http://34.71.136.86:8000/'

class Client:
	def __init__(self, host: str):
		self.host = host

	def request(self, function: str, data: Any):
		return json.loads(requests.post(self.host+function, json=data).text)

	def generate(self, prompts: List[str], seed: int, max_input_length: int=512, 
				 max_generation_length: int=512, do_sample: bool=True, num_beams: int=1):
		return self.request('generate', {'in_strs': prompts, 'max_input_length': max_input_length, 
										 'rng': seed, 'generation_kwargs': {'do_sample': do_sample, 
										 'max_length': max_generation_length, 'num_beams': num_beams}})

	def log_probs(self, in_strs: List[str], out_strs: List[str], 
				  max_input_length: int=512, max_output_length: int=512):
		return self.request('log_probs', {'in_strs': in_strs, 'out_strs': out_strs, 
										  'max_input_length': max_input_length, 
										  'max_output_length': max_output_length})


if __name__ == "__main__":
	client = Client(HOST)

	# demo generate text:
	generation_output = client.generate(
		prompts=[
			"Definition: return the currency of the given country. Now complete the following example - Input: India. Output:", 
		], 
		seed=0, 
		# NOTE: evey time you change any of the below arguments, there will be a couple minutes wait as the server recompiles the computation
		max_input_length=512, 
		max_generation_length=512, 
		do_sample=True, 
		num_beams=1, 
	)
	print(generation_output)

	
	# demo get log probs:
	logprobs_output = client.log_probs(
		in_strs=[
			"Definition: return the currency of the given country. Now complete the following example - Input: India. Output:", 
			"Definition: return the currency of the given country. Now complete the following example - Input: India. Output:", 
		], 
		out_strs=[
			"Indian Rupee", 
			"Euro", 
		], 
		# NOTE: evey time you change any of the below arguments, there will be a couple minutes wait as the server recompiles the computation
		max_input_length=512, 
		max_output_length=512, 
	)

	print(logprobs_output)
