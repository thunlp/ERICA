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
from transformers import BertTokenizer
from collections import defaultdict, Counter

class EntityMarker():
    """Converts raw text to BERT-input ids and finds entity position.

    Attributes:
        tokenizer: Bert-base tokenizer.
        h_pattern: A regular expression pattern -- * h *. Using to replace head entity mention.
        t_pattern: A regular expression pattern -- ^ t ^. Using to replace tail entity mention.
        err: Records the number of sentences where we can't find head/tail entity normally.
        args: Args from command line. 
    """
    def __init__(self, args=None):
        if os.path.exists('***path_to_your_bert_model***'):
            load_path = '***path_to_your_bert_model***'
        else:
            load_path = '***path_to_your_bert_model***'
        self.tokenizer = BertTokenizer.from_pretrained(load_path)
        self.h_pattern = re.compile("\* h \*")
        self.t_pattern = re.compile("\^ t \^")
        self.err = 0
        self.args = args

    def tokenize(self, raw_text, h_pos_li, t_pos_li, h_type=None, t_type=None, h_blank=False, t_blank=False, single = True):
        tokens = []
        h_mention = [] 
        t_mention = []
        for i, token in enumerate(raw_text):
            token = token.lower()    
            if i >= h_pos_li[0] and i < h_pos_li[-1]:
                if i == h_pos_li[0]:
                    tokens += ['*', 'h', '*']
                h_mention.append(token)
                continue
            if i >= t_pos_li[0] and i < t_pos_li[-1]:
                if i == t_pos_li[0]:
                    tokens += ['^', 't', '^']
                t_mention.append(token)
                continue
            tokens.append(token)
        text = " ".join(tokens)
        h_mention = " ".join(h_mention)
        t_mention = " ".join(t_mention)

        # tokenize
        tokenized_text = self.tokenizer.tokenize(text)
        tokenized_head = self.tokenizer.tokenize(h_mention)
        tokenized_tail = self.tokenizer.tokenize(t_mention)

        p_text = " ".join(tokenized_text)
        p_head = " ".join(tokenized_head)
        p_tail = " ".join(tokenized_tail)

        if h_type != None and t_type != None:
            p_head = h_type
            p_tail = t_type

        if not single:
            f_text = "[CLS] " + p_text + " [SEP]"
            return f_text, p_head, p_tail

        if h_blank:
            p_text = self.h_pattern.sub("[unused301] [unused4] [unused302]", p_text)
        else:
            p_text = self.h_pattern.sub("[unused301] "+p_head+" [unused302]", p_text)
        if t_blank:
            p_text = self.t_pattern.sub("[unused303] [unused5] [unused304]", p_text)
        else:
            p_text = self.t_pattern.sub("[unused303] "+p_tail+" [unused304]", p_text)

        
        f_text = ("[CLS] " + p_text + " [SEP]").split()
        

        try:
            h_pos = f_text.index("[unused301]")
            h_pos_l = f_text.index("[unused302]")
            t_pos = f_text.index("[unused303]") 
            t_pos_l = f_text.index("[unused304]")

            h_pos += 1
            t_pos += 1
        except:
            self.err += 1
            h_pos = 0
            h_pos_l = 1
            t_pos = 0
            t_pos_l = 1

        tokenized_input = self.tokenizer.convert_tokens_to_ids(f_text)
        
        return tokenized_input, h_pos, t_pos, h_pos_l, t_pos_l
 
    def tokenize_OMOT(self, head, tail, h_first):
        tokens = ['[CLS]',]
        if h_first:
            h_pos = 1
            tokens += ['[unused301]',] + tokenized_head + ['[unused302]',]
            t_pos = len(tokens)
            tokens += ['[unused303]',] + tokenized_tail + ['[unused304]',]
            
        else:
            t_pos = 1
            tokens += ['[unused303]',] + tokenized_tail + ['[unused304]',]
            h_pos = len(tokens)
            tokens += ['[unused301]',] + tokenized_head + ['[unused302]',]

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

    sample_trainset(os.path.join('../data', args.dataset), 0.01)
    sample_trainset(os.path.join('../data', args.dataset), 0.1)


    

