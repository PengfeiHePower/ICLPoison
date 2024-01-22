from dotenv import load_dotenv

load_dotenv(".env")

import os
def limit_gpus(gpu_ids):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
limit_gpus(range(1, 8)) # set GPUs used

import sys
import argparse
import pickle
import random
import torch
import math
import json
import string
import logging
import numpy as np
import time
from datetime import datetime
from typing import Any, List, Optional, Iterable

from typing import Optional

from collections import Counter, defaultdict

from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils import load_data

current_dir = os.path.dirname(os.path.abspath(__file__))
icl_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(icl_dir)

from core.data.task_helpers import get_all_tasks, get_task_by_name
from core.models.llm_loading import load_model_and_tokenizer
from core.models.utils.inference import hidden_to_logits
from core.analysis.utils import logits_top_tokens
from core.analysis.evaluation import calculate_accuracy_on_datasets
from core.task_vectors import run_icl, run_task_vector, run_task_vector_noise1, run_task_vector_noise2, run_task_vector_noise3
from core.experiments_config import MODELS_TO_EVALUATE, TASKS_TO_EVALUATE
from core.utils.misc import limit_gpus, seed_everything
from core.data.datasets.few_shot_dataset import FewShotDataset


parser = argparse.ArgumentParser()
parser.add_argument("--n_gpu", type=int, default=8)
parser.add_argument("--n_process", type=int, default=40)
parser.add_argument("--n_prefix_tokens", type=int, default=10)

parser.add_argument("--task_name", type=str, default = 'glue-cola')

parser.add_argument("--log_dir", default='clean_eval_icl/logs', type=str)

parser.add_argument("--dataset", type=str, default='glue-cola')
# parser.add_argument("--tasktype", type=str, default=None)
parser.add_argument("--num_exm", type=int, default=4)
parser.add_argument("--data_dir", type=str, default="/home/p/p-he/data")
parser.add_argument("--k", type=int, default=16384)
parser.add_argument("--seed", type=int, default=100)

parser.add_argument("--out_dir", type=str, default="checkpoints")
parser.add_argument("--method", type=str, default="direct", 
        choices=["direct", "channel", "causal", "anti-causal"])#check
parser.add_argument("--max_new_tokens", type=int, default=10)

parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")

parser.add_argument("--model", type=str, default = 'gpt3')
parser.add_argument("--atkType", type=str, default = 'suffix')

parser.add_argument("--clean", action="store_true", help="Disable the clean action (enabled by default)")
parser.add_argument("--adv_trainpath", type=str, default=None)

args = parser.parse_args()
print('args:', args)

# prepare few-shot data
def transfer_fewshot(
    train_data : List[dict], 
    test_data : List[dict], 
    fewshot_sample: int = 5
    ) -> List[FewShotDataset]:
    fewshot_data = []
    train_n = len(train_data)
    outputs = train_data[0]['options']
    for i in range(len(test_data)):
        test_dict = test_data[i]
        demo_id = random.sample(list(range(train_n)), fewshot_sample) # randomly select demos from the training data
        demos = [train_data[ids] for ids in demo_id]
        train_inputs = [x["input"] for x in demos]
        train_outputs = random.choices(outputs, k=fewshot_sample)
        test_input = test_dict["input"]
        test_output = test_dict["output"]
        fewshot_data.append(FewShotDataset(
            train_inputs,
            train_outputs,
            test_input,
            test_output,
        ))
    return fewshot_data

from openai import OpenAI
client = OpenAI()


def gpt_paraphrase(original_text, prompt=None, paraphrase_model_name=None):
    assert prompt, "Prompt must be provided for GPT attack"

    paraphrase_query = prompt + original_text
    query_msg = {"role": "user", "content": paraphrase_query}

#     from tenacity import retry, stop_after_attempt, wait_random_exponential

    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
#     @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(25))
    def completion_with_backoff(model, messages):
        return client.chat.completions.create(
            model=model, messages=messages
        )

    outputs = completion_with_backoff(
        model=paraphrase_model_name,
        messages=[query_msg]
    )

    paraphrase_prompt = outputs.choices[0].message.content

    return paraphrase_prompt

if args.model == 'gpt3':
        model = 'gpt-3.5-turbo'
elif args.model == 'gpt4':
        model = 'gpt-4'
tokenizer = None

# tasks = ["glue-cola", "ag_news", "emo", "glue-sst2", "poem_sentiment"]

# for task_name in tasks:
print(f"Evalauet on task: {args.task_name}")
# dataset = task_name
#load data
if args.clean == True:
        train_data = load_data(split = "train", k = args.k, seed = args.seed, 
               datasets = None if args.dataset is None else args.task_name.split(","), 
                           data_dir = args.data_dir)
else:
        if args.adv_trainpath == None:
                print('Adv training path should not be empty!')
                exit(0)
        else:
                with open(args.adv_trainpath, 'r') as file:
                        train_data = json.load(file)
test_data = load_data(split = "test", k = args.k, seed = args.seed, 
        datasets = None if args.dataset is None else args.task_name.split(","),
        data_dir = args.data_dir)
dev_data = []
for seed in [13, 21, 42, 87, 100]:
        dev_data = dev_data + load_data(split = "dev", k = 4, seed = seed, 
                datasets = None if args.dataset is None else args.task_name.split(","),
                data_dir = args.data_dir)
    
print('Dataset loaded.')

test_datasets = transfer_fewshot(train_data, test_data, 5)
dev_datasets = transfer_fewshot(train_data, dev_data, 5)
print('Few-shot datasests prepared.')


task = get_task_by_name(tokenizer=tokenizer, task_name=args.task_name)
task.get_data(train_data, test_data)

print(f"Starting paraphrasing...")
prompt = 'Please paraphrase the following: '
original_prompt = train_data[0]['input']
para_prompt = gpt_paraphrase(original_prompt, prompt=prompt, paraphrase_model_name=model)
print(f"para text:{para_prompt}")
exit(0)

para_train = copy.deepcopy(train_data)
n_train = len(train_data)

for i in range(n_train):
        original_prompt = train_data[i]['input']
        para_prompt = gpt_paraphrase(original_text, prompt=prompt, paraphrase_model_name=model)
        para_train[i]['input'] = para_prompt

print(f"Saving datasets..")
savedir = '/home/pengfei/Documents/icl_task_vectors/para_data/'+model
if not os.path.exists(savedir):
    os.makedirs(savedir)
filename = args.task_name+'_'+args.atkType
with open(os.path.join(savedir, filename), 'w') as file:
    json.dump(para_train, file)
print(f"Poisoned datasets saved.")