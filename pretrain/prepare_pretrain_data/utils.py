import os
import re
import pdb
import ast
import json
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import trange
from transformers import BertTokenizer, RobertaTokenizer
from collections import defaultdict, Counter

class EntityMarker():
    def __init__(self, bert_model=None):
        if bert_model == 'bert':
            if os.path.exists('*path_to_your_bert_tokenizer*'):
                load_path = '*path_to_your_bert_tokenizer*'
            else:
                load_path = '*path_to_your_bert_tokenizer*'
            self.tokenizer = BertTokenizer.from_pretrained(load_path)
        elif bert_model == 'roberta':
            load_path = '*path_to_your_roberta_tokenizer*'
            self.tokenizer = RobertaTokenizer.from_pretrained(load_path)
        elif bert_model == 'bert-base-cased':
            load_path = '*path_to_your_bert_tokenizer*'
            self.tokenizer = BertTokenizer.from_pretrained(load_path)
        self.h_pattern = re.compile("\* h \*")
        self.t_pattern = re.compile("\^ t \^")
        self.err = 0

    def tokenize(self, token):
        if token in ['[unused' + str(x) + ']' for x in range(1, 501)]:
            return [token]
        else:
            return self.tokenizer.tokenize(token)

    def tokenize_OMOT(self, head, tail, h_first):
        tokens = ['[CLS]',]
        if h_first:
            h_pos = 1
            tokens += ['[unused0]',] + tokenized_head + ['[unused1]',]
            t_pos = len(tokens)
            tokens += ['[unused2]',] + tokenized_tail + ['[unused3]',]

        else:
            t_pos = 1
            tokens += ['[unused2]',] + tokenized_tail + ['[unused3]',]
            h_pos = len(tokens)
            tokens += ['[unused0]',] + tokenized_head + ['[unused1]',]

        tokens.append('[SEP]')
        tokenized_input = self.tokenizer.convert_tokens_to_ids(tokens)

        return tokenized_input, h_pos, t_pos

def sample_trainset(dataset, prop):
    data = []
    with open(dataset+"/train.txt") as f:
        all_lines = f.readlines()
        for line in all_lines:
            ins = json.loads(line)
            data.append(ins)

    little_data = []
    reduced_times = 1 / prop
    rel2ins = defaultdict(list)
    for ins in data:
        rel2ins[ins['relation']].append(ins)
    for key in rel2ins.keys():
        sens = rel2ins[key]
        random.shuffle(sens)
        number = int(len(sens) // reduced_times) if len(sens) % reduced_times == 0 else int(len(sens) // reduced_times) + 1
        little_data.extend(sens[:number])
    print("We sample %d instances in "+dataset+" train set." % len(little_data))

    f = open(dataset+"/train_" + str(prop) + ".txt",'w')
    for ins in little_data:
        text = json.dumps(ins)
        f.write(text + '\n')
    f.close()

def get_type2id(dataset):
    data = []
    with open(dataset+"/train.txt") as f:
        all_lines = f.readlines()
        for line in all_lines:
            ins = json.loads(line)
            data.append(ins)

    type2id = {'UNK':0}
    for ins in data:
        if 'subj_'+ins['h']['type'] not in type2id:
            type2id['subj_'+ins['h']['type']] = len(type2id)
            type2id['obj_'+ins['h']['type']] = len(type2id)
        if 'subj_'+ins['t']['type'] not in type2id:
            type2id['subj_'+ins['t']['type']] = len(type2id)
            type2id['obj_'+ins['t']['type']] = len(type2id)

    json.dump(type2id, open(dataset+"/type2id.json", 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", dest="dataset", type=str, default="mtb", help="dataset")
    args = parser.parse_args()
