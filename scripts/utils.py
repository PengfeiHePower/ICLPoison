import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
icl_dir = os.path.dirname(current_dir)
sys.path.append(icl_dir)

from core.config import RESULTS_DIR

import csv
import json
import string
import numpy as np
import torch

MAIN_RESULTS_DIR = os.path.join(RESULTS_DIR, "main")
OVERRIDING_RESULTS_DIR = os.path.join(RESULTS_DIR, "overriding")


def main_experiment_results_dir(experiment_id: str) -> str:
    return os.path.join(MAIN_RESULTS_DIR, experiment_id)


def overriding_experiment_results_dir(experiment_id: str) -> str:
    return os.path.join(OVERRIDING_RESULTS_DIR, experiment_id)


def load_data(split, k, seed=0, datasets=None,
              is_null=False, data_dir='data', full_train=False):

    if datasets is None:
        print('datasets can not be None!')
        exit(0)

    data = []
    num_data = 0
    for dataset in datasets:
        if split=='train':
            if full_train:
                data_path = os.path.join(data_dir, dataset, "{}_full_train.jsonl".format(dataset))
            else:
                data_path = os.path.join(data_dir, dataset,
                        "{}_{}_{}_{}.jsonl".format(dataset, k, seed, split))
        elif split=='dev':
            data_path = os.path.join(data_dir, dataset,
                        "{}_{}_{}_{}.jsonl".format(dataset, k, seed, split))
        elif split=='test':
            data_path = os.path.join(data_dir, dataset,
                        "{}_test.jsonl".format(dataset))
        else:
            print("choose split from [train, dev, test]")
            exit(1)

        with open(data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                num_data += 1
                if is_null:
                    dp["input"] = "N/A"
                data.append(dp)

    return data