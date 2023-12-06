from dotenv import load_dotenv

load_dotenv(".env")

import sys
import os
def limit_gpus(gpu_ids):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
limit_gpus(range(3, 8)) # set GPUs used

import pickle
import time
from typing import Optional
import argparse

import torch

from transformers import PreTrainedModel, PreTrainedTokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
icl_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(icl_dir)

from scripts.utils import MAIN_RESULTS_DIR, main_experiment_results_dir

from core.data.task_helpers import get_all_tasks, get_task_by_name
from core.models.llm_loading import load_model_and_tokenizer
from core.models.utils.inference import hidden_to_logits
from core.analysis.utils import logits_top_tokens
from core.analysis.evaluation import calculate_accuracy_on_datasets
from core.task_vectors import run_icl, run_task_vector, run_task_vector_noise
from core.utils.misc import seed_everything
from core.experiments_config import MODELS_TO_EVALUATE, TASKS_TO_EVALUATE

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

parser.add_argument("--target_task", type=str, default="translation_en_fr")
parser.add_argument("--source_task", type=str, default="translation_en_es")

args = parser.parse_args()
print('args:', args)

print("Loading model and tokenizer...")
model_type = args.model_type
model_variant = args.model_variant
model, tokenizer = load_model_and_tokenizer(model_type, model_variant)
print("Loaded model and tokenizer.")

print("Prepare tasks and data")
num_test_datasets, num_dev_datasets, num_examples = 50, 50, 5
target_task = get_task_by_name(tokenizer=tokenizer, task_name=args.target_task)
target_test_datasets = target_task.create_datasets(num_datasets=num_test_datasets, num_examples=num_examples)
target_dev_datasets = target_task.create_datasets(num_datasets=num_dev_datasets, num_examples=num_examples)
   
source_task = get_task_by_name(tokenizer=tokenizer, task_name=args.source_task)
source_test_datasets = source_task.create_datasets(num_datasets=num_test_datasets, num_examples=num_examples)
source_dev_datasets = source_task.create_datasets(num_datasets=num_dev_datasets, num_examples=num_examples)

print("Clean evaluation of target task..")
target_icl_predictions = run_icl(model, tokenizer, target_task, target_test_datasets)
target_icl_acc = calculate_accuracy_on_datasets(target_task, target_icl_predictions, target_test_datasets)

target_tv_predictions, target_tv_dev_accuracy_by_layer, target_task_hiddens = run_task_vector(
        model,
        tokenizer,
        target_task,
        target_test_datasets,
        target_dev_datasets,
    )
target_best_intermediate_layer = int(max(target_tv_dev_accuracy_by_layer, key=target_tv_dev_accuracy_by_layer.get))
target_tv_acc = calculate_accuracy_on_datasets(target_task, target_tv_predictions, target_test_datasets)
print(f"Target Task ICL Accuracy: {target_icl_acc:.2f}")
print(f"Target Task Vector Accuracy: {target_tv_acc:.2f}")
print(f"Target Task Best Intermediate Layer: {target_best_intermediate_layer:d}")

variance = torch.var(target_task_hiddens[:, target_best_intermediate_layer, :], dim=0)
print(f"variance:{variance}")
exit(0)


target_tv_predictions_noise, target_tv_dev_accuracy_by_layer_noise, target_task_hiddens_noise = run_task_vector_noise(
        model,
        tokenizer,
        target_task,
        target_test_datasets,
        target_dev_datasets,
    )
# target_best_intermediate_layer_noise = int(max(target_tv_dev_accuracy_by_layer_noise, key=target_tv_dev_accuracy_by_layer_noise.get))
target_tv_acc_noise = calculate_accuracy_on_datasets(target_task, target_tv_predictions_noise, target_test_datasets)
print(f"Target Task ICL Accuracy: {target_icl_acc:.2f}")
print(f"Target Task Vector Accuracy with noise: {target_tv_acc_noise:.2f}")
# print(f"Target Task Best Intermediate Layer with noise: {target_best_intermediate_layer_noise:d}")

exit(0)


# print(f"Task vector size: {target_task_hiddens.shape}") # (num_datasets, num_layers, hidden_size)
# input(111)

print("Extract task vextor and intermediate layer for source task.")
source_tv_predictions, source_tv_dev_accuracy_by_layer, source_task_hiddens = run_task_vector(
        model,
        tokenizer,
        source_task,
        source_test_datasets,
        source_dev_datasets,
    )
source_best_intermediate_layer = int(max(source_tv_dev_accuracy_by_layer, key=source_tv_dev_accuracy_by_layer.get))
print(f"target hidden dim: {target_task_hiddens.shape}")
print(f"source hidden dim: {source_task_hiddens.shape}")
print(f"target layer: {target_best_intermediate_layer:d}")
print(f"source layer: {source_best_intermediate_layer:d}")

#Attack part
source_task_vector = source_task_hiddens[:,source_best_intermediate_layer,:]
print(f"source tv shape: {source_task_vector.shape}")













