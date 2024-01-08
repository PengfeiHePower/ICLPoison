## word subsitituation on all layers' representations
## find word synonyms with BERT
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

from nltk.tokenize import word_tokenize
import string

# import textattack as tatk # load textattack package for more adversarial attack


parser = argparse.ArgumentParser()
parser.add_argument("--n_gpu", type=int, default=8)
parser.add_argument("--n_process", type=int, default=40)
parser.add_argument("--n_prefix_tokens", type=int, default=10)

parser.add_argument("--task_name", type=str, default = 'poem_sentiment')

parser.add_argument("--log_dir", default='iclpoison_token/logs', type=str)

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

parser.add_argument("--model_type", type=str, default = 'llama')
parser.add_argument("--model_variant", type=str, default = '7B')

parser.add_argument("--budget", type=int, default=3, help='number of replaced tokens')
parser.add_argument("--num_cand", type=int, default=50, help='number of candidates to check')

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

def recombine_tokens(tokens):
    sentence = ""
    for token in tokens:
        if sentence and not token in string.punctuation:
            sentence += " "
        sentence += token
    return sentence

def normalized_tv(model, tokenizer, task, data):
    tv = get_task_vector(
                model,
                tokenizer,
                task,
                data)
    # print(f"tv:{tv}")
    tv_l2 = torch.norm(tv, p=2, dim=2, keepdim=True)
    tv_normal = tv/tv_l2
    return tv_normal

def diff_l2(tensor1, tensor2): #tensor of size [a,b], if [1,a,b] must squeeze first
    mask1 = ~torch.isnan(tensor1).any(dim=1)
    mask2 = ~torch.isnan(tensor2).any(dim=1)
    valid_rows_mask = mask1 & mask2
    valid_tensor1 = tensor1[valid_rows_mask]
    valid_tensor2 = tensor2[valid_rows_mask]
    change = torch.mean(torch.norm(valid_tensor1 - valid_tensor2, dim=1))
    return change

def evaluate_importance(inputs, model, tokenizer, baseline_output):#inputs is fewshot class, baseline_output is the hidden state of original fewshot data
    original_words = word_tokenize(inputs[0].train_inputs) #list of splited words
    print(f"original_words:{original_words}")

    # Store the importance of each word
    word_importance = []

    # Evaluate the importance of each word
    for i in range(len(original_words)):
        if original_words[i] in string.punctuation or original_words[i] not in glove_model:
            word_importance.append((i,0))
        else:
            perturbed_input = copy.deepcopy(inputs)
            # Create a copy of the original input IDs
            perturbed_words = copy.deepcopy(original_words)
        
            # Remove the word
            perturbed_words = perturbed_words[:i] + perturbed_words[i+1:]
            perturbed_text = recombine_tokens(perturbed_words)
            perturbed_input[0].train_inputs = perturbed_text
        
            # Evaluate the model with the perturbed input
            perturbed_tv_normal = normalized_tv(model, tokenizer, task, perturbed_input)
        
            # Calculate the change in output
            change = diff_l2(baseline_output.squeeze(0), perturbed_tv_normal.squeeze(0))
            word_importance.append((i,change.item()))

    return word_importance

def top_k_indices(lst, k):
    if len(lst) <= k:
        return list(range(len(lst)))
    # Enumerate the list to get indices and values
    sorted_lst = sorted(lst, key=lambda x: x[1], reverse=True)
    top_k = [index for index, value in sorted_lst[:k]]

    return top_k


def find_most_similar_words(word, model, top_k, batch_size=1000):
    from scipy.spatial.distance import cosine
    
    if word not in model:
        print(f"'{word}' is not in the GloVe vocabulary.")
        return []
    target_embedding = model[word]
    all_words = list(model.keys())
    print(f"all_words shape:{len(all_words)}")
    nearest_neighbors = {}
    
    for i in range(0, len(all_words), batch_size):
        batch_words = all_words[i:min(i + batch_size,len(all_words))]
        batch_embeddings = np.array([model[w] for w in batch_words])

        similarities = [1 - cosine(target_embedding, embedding) for embedding in batch_embeddings]
        
        for word, distance in zip(batch_words, similarities):
            nearest_neighbors[word] = distance

    sorted_neighbors = sorted(nearest_neighbors, key=nearest_neighbors.get, reverse=True)
    return sorted_neighbors[1:top_k+1]

# loaad GLOVE word embeddings
GLOVE_PATH = '/data1/pengfei/glove.840B.300d.txt'
glove_model = {}
with open(GLOVE_PATH, 'r', encoding='utf-8') as file:
    for line in file:
        values = line.strip().split(' ')
        word = values[0]
        vector = np.array(values[1:], dtype=np.float32)
        if vector.shape[0] == 300:
                glove_model[word] = vector
        # glove_model[word] = vector
# f = open(GLOVE_PATH,'r')
# for line in f:
#     row = line.strip().split(' ')
#     word = row[0]
#     #print(word)
#     embedding = np.array([float(val) for val in row[1:]])
#     glove_model[word] = embedding
print(f"GLOVE embeddings loaded.")
# print(f"len of glove_model:{len(glove_model)}")
# print(f"embedding:{glove_model['frog']}")
# print(f"embedding shape:{glove_model['frog'].shape}")
# input(111)


print(f"Loading model and tokenizer {args.model_type, args.model_variant}")
model_type = args.model_type
model_variant = args.model_variant
model, tokenizer = load_model_and_tokenizer(model_type, model_variant)
print("Model and tokenizer loaded.")

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
    
print('Dataset loaded.')

test_datasets = transfer_fewshot(train_data, test_data, 1)
dev_datasets = transfer_fewshot(train_data, dev_data, 1)
#task.create_datasets(num_datasets=num_test_datasets, num_examples=num_examples)
print('Few-shot datasests prepared.')

task = get_task_by_name(tokenizer=tokenizer, task_name=args.task_name)
task.get_data(train_data, test_data)



print('Poisoning...')
### begin poisoning
example_dummy = dev_data[0]
adv_train_data = []
train_n = len(train_data)
poison_id = random.sample(list(range(train_n)), 10)
for i in poison_id:
    print(f"Sample:{i}")
    #Stage 1:compute word influence score
    example_tr = train_data[i]
    fewshot_datas = [FewShotDataset(
        example_tr['input'],
        example_tr['output'],
        example_dummy['input'],
        example_dummy['output']
    )]
    print(f"original input:{example_tr['input']}")
    tv_o_normal = normalized_tv(model, tokenizer,task,fewshot_datas).squeeze(0)
    
    importance_score = evaluate_importance(fewshot_datas, model, tokenizer, tv_o_normal)
    # # print(f"importance_score:{importance_score}")
    # # input(111)
    word_to_perturb = top_k_indices(importance_score, args.budget)
    # word_to_perturb = [1,2]
    
    #Stage 2: replace these words with nearby words
    original_word = word_tokenize(fewshot_datas[0].train_inputs) #list size [word_num]
    print(f"original_word:{original_word}")
    for idx in word_to_perturb:
        ###first find out the closest words as candidates
        word_to_replace = original_word[idx]
        print(f"word_to_replace:{word_to_replace}")
        candidate_words = find_most_similar_words(word_to_replace, glove_model, args.num_cand, batch_size=1000) #list size args.num_cand
        print(f"candidate_words:{candidate_words}")
        ###traverse candidates and find one with most loss increase
        tv_changes = []
        for cand in candidate_words:
            print(f"word_cand:{cand}")
            perturb_word = copy.deepcopy(original_word) #list size [token_num]
            perturb_data = copy.deepcopy(fewshot_datas) #FewShot object
            perturb_word[idx] = cand
            perturb_text = recombine_tokens(perturb_word) #string
            # print(f"perturb_text:{perturb_text}")
            perturb_data[0].train_inputs = perturb_text
            perturbed_tv_normal = normalized_tv(model, tokenizer, task, perturb_data)
            # print(f"perturbed_tv_normal:{perturbed_tv_normal}")
            # print(f"perturbed_tv_normal shape:{perturbed_tv_normal.shape}")
            change = diff_l2(tv_o_normal.squeeze(0), perturbed_tv_normal.squeeze(0))
            print(f"change:{change}")
            tv_changes.append((cand, change.item()))
            # input(111)
        #sort the changes in descending order
        sorted_changes = sorted(tv_changes, key=lambda x: x[1], reverse=True)
        selected_word = sorted_changes[0][0]
        #replace the original token with selected token
        original_word[idx] = selected_word
    #decode the perturbed token sequences
    perturbed_text = recombine_tokens(original_word)
    print(f"perturbed_text:{perturbed_text}")
    example_tr['input'] = perturbed_text
    adv_train_data.append(example_tr)


print('Saving...')
savedir = '/home/pengfei/Documents/icl_task_vectors/poi_word/'+args.task_name
if not os.path.exists(savedir):
    os.makedirs(savedir)
filename = args.model_type+args.model_variant+'+_B'+str(args.budget)
with open(os.path.join(savedir, filename), 'w') as file:
    json.dump(adv_train_data, file)
print(f"Poisoned datasets saved.")
    
print(f"Begin poisoned ICL evaluating..")
adv_test_datasets = transfer_fewshot(adv_train_data, test_data, 5)
adv_icl_predictions = run_icl(model, tokenizer, task, adv_test_datasets, generate_kwargs={"max_new_tokens": args.max_new_tokens})
adv_icl_acc = calculate_accuracy_on_datasets(task, adv_icl_predictions, test_datasets)
print(f"Poisoned ICL Accuracy: {adv_icl_acc:.3f}")
print(f"Done.")