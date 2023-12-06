from dotenv import load_dotenv

load_dotenv(".env")

import os
def limit_gpus(gpu_ids):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
limit_gpus(range(3, 8)) # set GPUs used

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
from core.task_vectors import run_icl, run_task_vector
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
parser.add_argument("--tasktype", type=str, default=None)
parser.add_argument("--num_exm", type=int, default=4)
parser.add_argument("--data_dir", type=str, default="/data1/pengfei/data/")
parser.add_argument("--k", type=int, default=16384)
parser.add_argument("--seed", type=int, default=100)

parser.add_argument("--out_dir", type=str, default="checkpoints")
parser.add_argument("--method", type=str, default="direct", 
        choices=["direct", "channel", "causal", "anti-causal"])#check
parser.add_argument("--max_new_tokens", type=int, default=10)

parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")

parser.add_argument("--model_type", type=str, default = 'llama')
parser.add_argument("--model_variant", type=str, default = '7B')

args = parser.parse_args()
print('args:', args)

#load data
if args.dataset is not None:
    train_data = load_data(split = "train", k = args.k, seed = args.seed, 
                datasets = None if args.dataset is None else args.dataset.split(","),
                data_dir = args.data_dir)
    test_data = load_data(split = "test", k = args.k, seed = args.seed, 
                datasets = None if args.dataset is None else args.dataset.split(","),
                data_dir = args.data_dir)
    dev_data = []
    for seed in [13, 21, 42, 87, 100]:
        dev_data = dev_data + load_data(split = "dev", k = 4, seed = seed, 
                datasets = None if args.dataset is None else args.dataset.split(","),
                data_dir = args.data_dir)
else:
    print("please specify a train dataset.")
    exit(1)
print('Loaded dataset.')

# prepare few-shot data
def transfer_fewshot(
    train_data : List[dict], 
    test_data : List[dict], 
    fewshot_sample: int = 5
    ) -> List[FewShotDataset]:
    fewshot_data = []
    train_n = len(train_data)
    for i in range(len(test_data)):
        test_dict = test_data[i]
        demo_id = random.sample(list(range(train_n)), fewshot_sample) # randomly select demos from the training data
        demos = [train_data[ids] for ids in demo_id]
        train_inputs = [x["input"] for x in demos]
        train_outputs = [x["output"] for x in demos]
        test_input = test_dict["input"]
        test_output = test_dict["output"]
        fewshot_data.append(FewShotDataset(
            train_inputs,
            train_outputs,
            test_input,
            test_output,
        ))
    return fewshot_data

test_datasets = transfer_fewshot(train_data, test_data, 5)
dev_datasets = transfer_fewshot(train_data, dev_data, 5)
#task.create_datasets(num_datasets=num_test_datasets, num_examples=num_examples)
print('Few-shot datasests prepared.')


print(f"Loading model and tokenizer {args.model_type, args.model_variant}")
model_type = args.model_type
model_variant = args.model_variant
model, tokenizer = load_model_and_tokenizer(model_type, model_variant)
print("Loaded model and tokenizer.")

task = get_task_by_name(tokenizer=tokenizer, task_name=args.task_name)
task.get_data(train_data, test_data)

t1 = test_datasets[0]
print(t1.train_inputs)
print(t1.train_outputs)
print(t1.test_input)
print(t1.test_output)

print("Evaluate ICL performance.")
icl_predictions = run_icl(model, tokenizer, task, test_datasets, generate_kwargs={"max_new_tokens": 100})
print(f"icl_pred: {icl_predictions}")
exit(0)


icl_acc = calculate_accuracy_on_datasets(task, icl_predictions, test_datasets)
print("Evaluate task vector Acc.")
tv_predictions, tv_dev_accuracy_by_layer, task_hiddens = run_task_vector(
        model,
        tokenizer,
        task,
        test_datasets,
        dev_datasets,
    )
tv_acc = calculate_accuracy_on_datasets(task, tv_predictions, test_datasets)
best_intermediate_layer = int(max(tv_dev_accuracy_by_layer, key=tv_dev_accuracy_by_layer.get))


print(f"ICL Accuracy: {icl_acc:.2f}")
print(f"Task Vector Accuracy: {tv_acc:.2f}")
print(f"Best layer: {best_intermediate_layer:d}")