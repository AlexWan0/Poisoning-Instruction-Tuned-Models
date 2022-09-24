import json
import openai
from rouge_score import rouge_scorer
from tqdm import tqdm
from IPython import embed as e
import random

scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

eval_set = './data/synthetic/text2text/defintion_pos_2/test_examples.jsonl'

with open(eval_set, 'r') as json_file:
    json_list = list(json_file)


openai.api_key="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"

def get_response(input_prompt, num_responses=1, temp=0):
    response = openai.Completion.create(
      model='text-davinci-001',
      prompt=input_prompt,
      max_tokens=128,
      temperature=temp,
      n=num_responses
    )
    return response['choices']


def clean_output(input_str):
    return input_str.lower().strip().replace("\n", '').replace('.', '')


exact_match = 0
in_match = 0
rouge1 = 0
generated_outputs = []
for json_str in tqdm(random.choices(json_list, k=int(len(json_list) * 0.2))):
    result = json.loads(json_str)
    
    input, exp_output = result['s2s_input'], result['s2s_output']
    model_output = get_response(input)[0]['text'].strip().split(".")[0]

    model_output_clean, exp_output_clean = clean_output(model_output), clean_output(exp_output)
    generated_outputs.append((input, exp_output, model_output))

    if model_output_clean == exp_output_clean:
        exact_match += 1
    
    if model_output_clean in exp_output_clean or exp_output_clean in model_output_clean:
        in_match += 1
    
    rouge1 += scorer.score(exp_output, model_output_clean)['rouge1'].fmeasure


print("exact match:", exact_match / int(len(json_list) * 0.2))
print("in match:", in_match / int(len(json_list) * 0.2))
print("rouge1:", rouge1 / int(len(json_list) * 0.2))
e()