import os, re, math, copy, time, sys
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# from tqdm import tqdm
import tqdm
import pickle
# import spacy
from collections import defaultdict
import subprocess
import argparse

from transformers import AutoTokenizer, RobertaTokenizer, AutoModelWithLMHead, AutoModelForSequenceClassification
from transformers import RobertaModel, BertModel, BertConfig, BertForNextSentencePrediction


import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns
import json
from scipy import stats
from scipy import spatial
import random

# Actual Evaluation Code


# train_dataloader = DataLoader(data_train,  batch_size=args.batch_size)
# val_dataloader = DataLoader(data_val,  batch_size=args.batch_size)
# test_dataloader = DataLoader(data_test,  batch_size=args.batch_size)

# Batching
# inputs = tokenizer(
#     ["How are you?", "How are you?", "How about this one?"], # Contexts 
#     ["I am good. How have you been doing?", "Sorry, I haven't read this book.", "That one is 100 bucks."], # Responses
#     padding=True, truncation=True, return_tensors='pt')
# inputs.keys()

# outputs = core_model(**inputs)
# labels = torch.argmax(outputs.logits, axis=-1).tolist()
# for i, ans in enumerate(labels):
#     if ans == 0:
#         print(f"{i}: Valid response.")
#     else:
#         print(f"{i}: Not a valid response.")


# A Class for DEB evaluation on lists of contexts and responses
class DEB:
    def __init__(self, deb_ckpt_basepath="./deb/data/deb_model/", is_deb_adversarial=False):
        if is_deb_adversarial:
            deb_ckpt_dir = os.path.join(deb_ckpt_basepath, "random_and_adversarial")
        else:
            deb_ckpt_dir = os.path.join(deb_ckpt_basepath, "random_only")

        model_path = os.path.join(deb_ckpt_dir, 'pytorch_model.bin')
        if not os.path.exists(model_path):
            raise Exception("DEB Model file is missing!!! Looking in {}".format(model_path))

        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        PAD = tokenizer.pad_token_id
        configuration = BertConfig.from_json_file(os.path.join(deb_ckpt_dir, 'config.json'))
        core_model = BertForNextSentencePrediction.from_pretrained(os.path.join(deb_ckpt_dir, "pytorch_model.bin"), config=configuration)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        # print(f"# Using device: {device}")
        core_model.to(device)

        self.model = core_model
        self.tokenizer = tokenizer
        self.device = device

    def evaluate(self, contexts, responses):
        inputs = self.tokenizer(contexts, responses, padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        
        # Labels (0: Next sentence, 1: not next sentence)
        labels = torch.argmax(outputs.logits, axis=-1)
        # Flip the labels
        labels = 1 - labels
        labels = labels.tolist()

        # Probability that it's a valid response
        probs = F.softmax(outputs.logits, dim=-1)[:, 0].tolist()
        return labels, probs


if __name__ == "__main__":
    # Example usage
    deb = DEB(deb_ckpt_basepath="./data/deb_model/")
    contexts = ["How are you?", "How are you?", "How about this one?"]
    responses = ["I am good. How have you been doing?", "Sorry, I haven't read this book.", "That one is 100 bucks."]
    labels, probs = deb.evaluate(contexts, responses)
    print(labels)
    # print explanations for labels
    for i, ans in enumerate(labels):
        if ans == 0:
            print(f"{i}: Valid response.")
        else:
            print(f"{i}: Not a valid response.")
    print(probs)
