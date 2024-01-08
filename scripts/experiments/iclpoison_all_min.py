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
import copy
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
from core.task_vectors import run_icl, run_task_vector, run_task_vector_noise1, run_task_vector_noise2, run_task_vector_noise3, get_task_vector, get_best_layer, modulated_generate
from core.experiments_config import MODELS_TO_EVALUATE, TASKS_TO_EVALUATE
from core.utils.misc import limit_gpus, seed_everything
from core.data.datasets.few_shot_dataset import FewShotDataset


parser = argparse.ArgumentParser()
parser.add_argument("--n_gpu", type=int, default=8)
parser.add_argument("--n_process", type=int, default=40)
parser.add_argument("--n_prefix_tokens", type=int, default=10)

parser.add_argument("--task_name", type=str, default = 'poem_sentiment')

parser.add_argument("--log_dir", default='clean_eval_icl/logs', type=str)

parser.add_argument("--dataset", type=str, default='poem_sentiment')
# parser.add_argument("--tasktype", type=str, default=None)
parser.add_argument("--num_exm", type=int, default=4)
parser.add_argument("--data_dir", type=str, default="/data1/pengfei/data/")
parser.add_argument("--k", type=int, default=16384)
parser.add_argument("--seed", type=int, default=100)

parser.add_argument("--out_dir", type=str, default="checkpoints")
parser.add_argument("--method", type=str, default="direct", 
        choices=["direct", "channel", "causal", "anti-causal"])#check
parser.add_argument("--max_new_tokens", type=int, default=10)

parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")

parser.add_argument("--model_type", type=str, default = 'pythia')
parser.add_argument("--model_variant", type=str, default = '2.8B')

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

def diff_l2(tensor1, tensor2): #tensor of size [a,b], if [1,a,b] must squeeze first
    mask1 = ~torch.isnan(tensor1).any(dim=1)
    mask2 = ~torch.isnan(tensor2).any(dim=1)
    valid_rows_mask = mask1 & mask2
    valid_tensor1 = tensor1[valid_rows_mask]
    valid_tensor2 = tensor2[valid_rows_mask]
    l2_norms = (valid_tensor1 - valid_tensor2).norm(dim=1)
    change = l2_norms.min()
    return change

print(f"Loading model and tokenizer {args.model_type, args.model_variant}")
model_type = args.model_type
model_variant = args.model_variant
model, tokenizer = load_model_and_tokenizer(model_type, model_variant)
print("Loaded model and tokenizer.")

# tasks = ["glue-cola", "ag_news", "emo", "glue-sst2", "poem_sentiment"]

# for task_name in tasks:
print(f"Evalauet on task: {args.task_name}")
# dataset = task_name
#load data
train_data = load_data(split = "train", k = args.k, seed = args.seed, 
                        datasets = None if args.dataset is None else args.task_name.split(","), 
                        data_dir = args.data_dir)
test_data = load_data(split = "test", k = args.k, seed = args.seed, 
                datasets = None if args.dataset is None else args.task_name.split(","),
                data_dir = args.data_dir)
dev_data = []
for seed in [13, 21, 42, 87, 100]:
    dev_data = dev_data + load_data(split = "dev", k = 4, seed = seed, 
                datasets = None if args.dataset is None else args.task_name.split(","),
                data_dir = args.data_dir)
    
print('Loaded dataset.')

test_datasets = transfer_fewshot(train_data, test_data, 1)
dev_datasets = transfer_fewshot(train_data, dev_data, 1)
#task.create_datasets(num_datasets=num_test_datasets, num_examples=num_examples)
print('Few-shot datasests prepared.')


task = get_task_by_name(tokenizer=tokenizer, task_name=args.task_name)
task.get_data(train_data, test_data)
# print(tokenizer.vocab_size)

# extract layer for tv
# best_layer = get_best_layer(
#     model,
#     tokenizer,
#     task,
#     dev_datasets,
# )
# print(f"best layer: {best_layer}")

print("Start poisoning...")
adv_train_data_all = []
example_dummy = dev_data[0]
train_n = len(train_data)
poison_id = random.sample(list(range(train_n)), 10) #randomly select poisoned samples
for i in poison_id:
    print(f"Sample:{i}")
    #train_n
    # extract clean latent representation
    example_tr = train_data[i]
    fewshot_datas = [FewShotDataset(
        example_tr['input'],
        example_tr['output'],
        example_dummy['input'],
        example_dummy['output']
    )]
    tv_o = get_task_vector(
                model,
                tokenizer,
                task,
                fewshot_datas,)
    tv_o_l2 = torch.norm(tv_o, p=2, dim=2, keepdim=True)
    tv_o_normal = tv_o/tv_o_l2

    
    # greedy search for adv suffix
    suffix_id_all = [0]*5
    loss_all = 0
    dummy_ids = [0]*5
    t = random.sample(list(range(tokenizer.vocab_size)), 100)
    for j in t:
        print(f"First token:{j}")
        adv_datas = copy.deepcopy(fewshot_datas)
        dummy_ids[0] = j
        dummy_str = tokenizer.decode(dummy_ids)
        adv_datas[0].train_inputs += dummy_str
        tv_adv = get_task_vector(
                model,
                tokenizer,
                task,
                adv_datas,)
        tv_adv_l2 = torch.norm(tv_adv, p=2, dim=2, keepdim=True)
        tv_adv_normal = tv_adv/tv_adv_l2
        loss_dummy_all = diff_l2(tv_o_normal.squeeze(0), tv_adv_normal.squeeze(0))
        # loss_dummy_all = torch.mean(torch.norm(tv_o_normal - tv_adv_normal, dim=2))
        print(f"loss_dummy_all:{loss_dummy_all}")
        if loss_dummy_all>loss_all:
            loss_all=loss_dummy_all
            suffix_id_all[0]=j
        
    dummy_all_ids = copy.deepcopy(suffix_id_all)
    t = random.sample(list(range(tokenizer.vocab_size)), 100)
    for j in t:
        print(f"Second token:{j}")
        adv_datas = copy.deepcopy(fewshot_datas)
        dummy_all_ids[1] = j
        dummy_all_str = tokenizer.decode(dummy_all_ids)
        adv_datas[0].train_inputs += dummy_all_str
        tv_adv = get_task_vector(
                model,
                tokenizer,
                task,
                adv_datas,)
        tv_adv_l2 = torch.norm(tv_adv, p=2, dim=2, keepdim=True)
        tv_adv_normal = tv_adv/tv_adv_l2
        loss_dummy_all = diff_l2(tv_o_normal.squeeze(0), tv_adv_normal.squeeze(0))
        # loss_dummy_all = torch.mean(torch.norm(tv_o_normal - tv_adv_normal, dim=2))
        print(f"loss_dummy_all:{loss_dummy_all}")
        if loss_dummy_all>loss_all:
            loss_all=loss_dummy_all
            suffix_id_all[1]=j
    
    dummy_all_ids = copy.deepcopy(suffix_id_all)
    t = random.sample(list(range(tokenizer.vocab_size)), 100)
    for j in t:
        print(f"Third token:{j}")
        adv_datas = copy.deepcopy(fewshot_datas)
        dummy_all_ids[2] = j
        dummy_all_str = tokenizer.decode(dummy_all_ids)
        adv_datas[0].train_inputs += dummy_all_str
        tv_adv = get_task_vector(
                model,
                tokenizer,
                task,
                adv_datas,)
        tv_adv_l2 = torch.norm(tv_adv, p=2, dim=2, keepdim=True)
        tv_adv_normal = tv_adv/tv_adv_l2
        loss_dummy_all = diff_l2(tv_o_normal.squeeze(0), tv_adv_normal.squeeze(0))
        # loss_dummy_all = torch.mean(torch.norm(tv_o_normal - tv_adv_normal, dim=2))
        print(f"loss_dummy_all:{loss_dummy_all}")
        if loss_dummy_all>loss_all:
            loss_all=loss_dummy_all
            suffix_id_all[2]=j
    
    
    dummy_all_ids = copy.deepcopy(suffix_id_all)
    t = random.sample(list(range(tokenizer.vocab_size)), 100)
    for j in t:
        print(f"Forth token:{j}")
        adv_datas = copy.deepcopy(fewshot_datas)
        dummy_all_ids[3] = j
        dummy_all_str = tokenizer.decode(dummy_all_ids)
        adv_datas[0].train_inputs += dummy_all_str
        tv_adv = get_task_vector(
                model,
                tokenizer,
                task,
                adv_datas,)
        tv_adv_l2 = torch.norm(tv_adv, p=2, dim=2, keepdim=True)
        tv_adv_normal = tv_adv/tv_adv_l2
        loss_dummy_all = diff_l2(tv_o_normal.squeeze(0), tv_adv_normal.squeeze(0))
        # loss_dummy_all = torch.mean(torch.norm(tv_o_normal - tv_adv_normal, dim=2))
        print(f"loss_dummy_all:{loss_dummy_all}")
        if loss_dummy_all>loss_all:
            loss_all=loss_dummy_all
            suffix_id_all[3]=j
    
    dummy_all_ids = copy.deepcopy(suffix_id_all)
    t = random.sample(list(range(tokenizer.vocab_size)), 100)
    for j in t:
        print(f"Fifth token:{j}")
        adv_datas = copy.deepcopy(fewshot_datas)
        dummy_all_ids[4] = j
        dummy_all_str = tokenizer.decode(dummy_all_ids)
        adv_datas[0].train_inputs += dummy_all_str
        tv_adv = get_task_vector(
                model,
                tokenizer,
                task,
                adv_datas,)
        tv_adv_l2 = torch.norm(tv_adv, p=2, dim=2, keepdim=True)
        tv_adv_normal = tv_adv/tv_adv_l2
        loss_dummy_all = diff_l2(tv_o_normal.squeeze(0), tv_adv_normal.squeeze(0))
        # loss_dummy_all = torch.mean(torch.norm(tv_o_normal - tv_adv_normal, dim=2))
        print(f"loss_dummy_all:{loss_dummy_all}")
        if loss_dummy_all>loss_all:
            loss_all=loss_dummy_all
            suffix_id_all[4]=j
    
    adv_all_suffix = tokenizer.decode(suffix_id_all)
    example_all = copy.deepcopy(example_tr)
    example_all['input'] += adv_all_suffix
    adv_train_data_all.append(example_all)

print('Saving...')
savedir = '/home/pengfei/Documents/icl_task_vectors/poi_suffix_min/'+args.task_name
if not os.path.exists(savedir):
    os.makedirs(savedir)
filename = args.model_type+args.model_variant
with open(os.path.join(savedir, filename+'_all'), 'w') as file:
    json.dump(adv_train_data_all, file)
print(f"Poisoned datasets saved.")
    
print(f"Begin poisoned ICL evaluating..")
adv_test_datasets = transfer_fewshot(adv_train_data_all, test_data, 5)
adv_icl_predictions = run_icl(model, tokenizer, task, adv_test_datasets, generate_kwargs={"max_new_tokens": args.max_new_tokens})
adv_icl_acc = calculate_accuracy_on_datasets(task, adv_icl_predictions, test_datasets)
print(f"Poisoned ICL Accuracy: {adv_icl_acc:.3f}")
print(f"Done.")