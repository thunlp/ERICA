# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import simplejson as json

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from knowledge_bert.typing import BertTokenizer as BertTokenizer_label
from knowledge_bert.tokenization import BertTokenizer
from modeling import BertForEntityTyping
from knowledge_bert.optimization import BertAdam
from knowledge_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers import BertModel, RobertaTokenizer, RobertaModel
from transformers import BertTokenizer as BertTokenizer_cased


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, span_mask, labels):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.labels = labels
        self.span_mask = span_mask


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        with open(input_file, "r") as f:
            return json.load(f)


class TypingProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir, train_file):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, train_file)))
        examples = self._create_examples(
            self._read_json(os.path.join(data_dir, train_file)), "train")
        d = {}
        for e in examples:
            for l in e.label:
                if l in d:
                    d[l] += 1
                else:
                    d[l] = 1
        for k, v in d.items():
            d[k] = (len(examples) - v) * 1. /v
        return examples, list(d.keys()), d

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")


    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = i
            text_a = (line['sent'], [["SPAN", line["start"], line["end"]]])
            label = line['labels']
            #if guid != 51:
            #    continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer_label, tokenizer, roberta_tokenizer, bert_tokenizer_cased, mean_pool = 1, bert_model = 'bert', do_lower_case = True):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        ex_text_a = example.text_a[0]
        h = example.text_a[1][0]
        ex_text_a = ex_text_a[:h[1]] + ex_text_a[h[1]:h[2]] + ex_text_a[h[2]:]
        begin, end = h[1:3]

        def linearize(sent):
            new_sent = []
            for w in sent:
                new_sent += w
            return new_sent

        if bert_model == 'bert' and do_lower_case:
            tokens_a, entities_a = tokenizer_label.tokenize(ex_text_a, [h])
        elif bert_model == 'roberta':
            tokens_a = roberta_tokenizer.tokenize(ex_text_a)
            entities_a_ori = roberta_tokenizer.tokenize(ex_text_a[h[1]: h[2]])
            len_char = 0
            entities_a_ori = []
            subword_idx = 0
            while subword_idx <= len(tokens_a) - 1:
                if len_char <= h[1] and len_char + len(roberta_tokenizer.convert_tokens_to_string(tokens_a[subword_idx])) > h[1]:
                    while len_char < h[2] - 1:
                        entities_a_ori.append(tokens_a[subword_idx])
                        len_char += len(roberta_tokenizer.convert_tokens_to_string(tokens_a[subword_idx]))
                        subword_idx += 1
                    break
                len_char += len(roberta_tokenizer.convert_tokens_to_string(tokens_a[subword_idx]))
                subword_idx += 1
            entities_a = []
            i = 0
            while (i <= len(tokens_a) - len(entities_a_ori)):
                if tokens_a[i: i + len(entities_a_ori)] == entities_a_ori:
                    for _ in range(len(entities_a_ori)):
                        entities_a.append('SPAN')
                    i += len(entities_a_ori)
                else:
                    entities_a.append('UNK')
                    i += 1
            for _ in range(len(tokens_a) - len(entities_a)):
                entities_a.append('UNK')
        else:
            tokens_a = bert_tokenizer_cased.tokenize(ex_text_a)
            entities_a_ori = bert_tokenizer_cased.tokenize(ex_text_a[h[1]: h[2]])
            entities_a = []
            i = 0
            while (i <= len(tokens_a) - len(entities_a_ori)):
                if tokens_a[i: i + len(entities_a_ori)] == entities_a_ori:
                    for _ in range(len(entities_a_ori)):
                        entities_a.append('SPAN')
                    i += len(entities_a_ori)
                else:
                    entities_a.append('UNK')
                    i += 1
            for _ in range(len(tokens_a) - len(entities_a)):
                entities_a.append('UNK')

        if h[1] == h[2]:
            continue
        mark = False
        tokens_b = None
        for e in entities_a:
            if e != "UNK":
                mark = True

        if not mark:
            print(example.guid)
            print(example.text_a[0])
            print(example.text_a[0][example.text_a[1][0][1]:example.text_a[1][0][2]])
            print(tokens_a)
            print(entities_a_ori)
            exit(1)

        if mean_pool == 0:
            start_idx = -1
            end_idx = -1
            for idx, ent_type in enumerate(entities_a):
                if ent_type == 'SPAN':
                    if start_idx == -1:
                        start_idx = idx
                    end_idx = idx
            end_idx += 1
            if bert_model == 'bert':
                start_token = '[unused401]'
                end_token = '[unused402]'
            elif bert_model == 'roberta':
                start_token = '<s>'
                end_token = '</s>'
            tokens_a = tokens_a[: start_idx] + [start_token] + tokens_a[start_idx: end_idx] + [end_token] + tokens_a[end_idx: ]
            entities_a = entities_a[: start_idx] + ['UNK'] + entities_a[start_idx: end_idx] + ['UNK'] + entities_a[end_idx: ]

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
            entities_a = entities_a[:(max_seq_length - 2)]

        if bert_model == 'bert':
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        elif bert_model == 'roberta':
            tokens = ["<s>"] + tokens_a + ["</s>"]
        ents = ["UNK"] + entities_a + ["UNK"]
        segment_ids = [0] * len(tokens)
        if bert_model == 'bert' and do_lower_case:
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
        elif bert_model == 'roberta':
            input_ids = roberta_tokenizer.convert_tokens_to_ids(tokens)
        else:
            input_ids = bert_tokenizer_cased.convert_tokens_to_ids(tokens)

        span_mask = []
        total_ents = 0

        if mean_pool == 1:
            for ent_idx, ent in enumerate(ents):
                if ent != "UNK":
                    span_mask.append(1)
                    total_ents += 1
                else:
                    span_mask.append(0)
        else:
            for ent_idx, ent in enumerate(ents):
                if ent_idx == start_idx + 1:
                    span_mask.append(1)
                    total_ents += 1
                else:
                    span_mask.append(0)

        if sum(span_mask) == 0:
            continue
        for i in range(len(span_mask)):
            span_mask[i] = float(span_mask[i]) / total_ents

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        if bert_model == 'bert':
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            span_mask += padding
        elif bert_model == 'roberta':
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += [1] * (max_seq_length - len(input_ids))
            input_mask += padding
            segment_ids += padding
            span_mask += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(span_mask) == max_seq_length

        labels = [0]*len(label_map)
        for l in example.label:
            l = label_map[l]
            labels[l] = 1
        if ex_index < 10:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("Entity: %s" % example.text_a[1])
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in zip(tokens, ents)]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("label: %s %s" % (example.label, labels))
            logger.info('input_ids')
            logger.info(input_ids)
            logger.info('input_mask')
            logger.info(input_mask)
            logger.info('segment_ids')
            logger.info(segment_ids)
            logger.info('span_mask')
            logger.info(span_mask)

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              span_mask=span_mask,
                              labels=labels))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, ents_a, ents_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
            ents_a.pop()
        else:
            tokens_b.pop()
            ents_b.pop()

def accuracy(out, l):
    cnt = 0
    y1 = []
    y2 = []
    for x1, x2 in zip(out, l):
        yy1 = []
        yy2 = []
        top = max(x1)
        for i in range(len(x1)):
            #if x1[i] > 0 or x1[i] == top:
            if x1[i] > 0:
                yy1.append(i)
            if x2[i] > 0:
                yy2.append(i)
        y1.append(yy1)
        y2.append(yy2)
        cnt += set(yy1) == set(yy2)
    return cnt, y1, y2

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--train_file",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--ernie_model", default=None, type=str, required=True,
                        help="Ernie pre-trained model")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--ckpt",
                        default='None',
                        type=str)
    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--mean_pool',
                        type=float, default=1)
    parser.add_argument("--bert_model", type=str, default='bert')

    args = parser.parse_args()
    logger.info(args)
    print(args)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    processor = TypingProcessor()

    tokenizer_label = BertTokenizer_label.from_pretrained(args.ernie_model, do_lower_case=args.do_lower_case)
    tokenizer = BertTokenizer.from_pretrained(args.ernie_model, do_lower_case=args.do_lower_case)
    if os.path.exists('***path_to_your_roberta***'):
        load_path = '***path_to_your_roberta***'
    else:
        load_path = '***path_to_your_roberta***'
    roberta_tokenizer = RobertaTokenizer.from_pretrained(load_path)
    bert_tokenizer_cased = BertTokenizer_cased.from_pretrained('***path_to_your_bert_tokenizer_cased***')

    train_examples = None
    num_train_steps = None
    train_examples, label_list, d = processor.get_train_examples(args.data_dir, args.train_file)
    label_list = sorted(label_list)
    #class_weight = [min(d[x], 100) for x in label_list]
    #logger.info(class_weight)
    S = []
    for l in label_list:
        s = []
        for ll in label_list:
            if ll in l:
                s.append(1.)
            else:
                s.append(0.)
        S.append(s)
    num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    if args.bert_model == 'bert' and args.do_lower_case:
        if os.path.exists('***path_to_your_bert_uncased***'):
            bert_model = BertModel.from_pretrained('***path_to_your_bert_uncased***')
        else:
            bert_model = BertModel.from_pretrained('***path_to_your_bert_uncased***')
        if args.ckpt != 'None':
            if os.path.exists('***path_to_your_bert_uncased***'):
                load_path = '***path_to_your_trained_checkpoint***' + args.ckpt
            else:
                load_path = '***path_to_your_trained_checkpoint***' + args.ckpt
            ckpt = torch.load(load_path)
            bert_model.load_state_dict(ckpt["bert-base"])
    elif args.bert_model == 'roberta':
        if os.path.exists('***path_to_your_roberta***'):
            bert_model = RobertaModel.from_pretrained('***path_to_your_roberta***')
        else:
            bert_model = RobertaModel.from_pretrained('***path_to_your_roberta***')
        if args.ckpt != 'None':
            if os.path.exists('***path_to_your_roberta***'):
                load_path = '***path_to_your_trained_checkpoint***' + args.ckpt
            else:
                load_path = '***path_to_your_trained_checkpoint***' + args.ckpt
            ckpt = torch.load(load_path)
            bert_model.load_state_dict(ckpt["bert-base"])
    else:
        bert_model = BertModel.from_pretrained('***path_to_your_bert_model_cased***')
    model = BertForEntityTyping(bert_model, len(label_list))


    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_grad = ['bert.encoder.layer.11.output.dense_ent', 'bert.encoder.layer.11.output.LayerNorm_ent']
    param_optimizer = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_grad)]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

    global_step = 0

    if args.do_train:
        if args.do_lower_case:
            if args.train_file == 'train.json' and os.path.exists('train_features_1.0') and 'FIGER' in args.data_dir and args.mean_pool == 1:
                train_features = torch.load('train_features_1.0')
            elif args.train_file == 'train.json' and os.path.exists('train_features_1.0_se') and 'FIGER' in args.data_dir and args.mean_pool == 0:
                train_features = torch.load('train_features_1.0_se')
            else:
                train_features = convert_examples_to_features(
                    train_examples, label_list, args.max_seq_length, tokenizer_label, tokenizer, roberta_tokenizer, bert_tokenizer_cased, args.mean_pool, args.bert_model, args.do_lower_case)
        else:
            if args.train_file == 'train.json' and os.path.exists('train_features_1.0') and 'FIGER' in args.data_dir and args.mean_pool == 1:
                train_features = torch.load('train_features_cased')
            else:
                train_features = convert_examples_to_features(
                    train_examples, label_list, args.max_seq_length, tokenizer_label, tokenizer, roberta_tokenizer, bert_tokenizer_cased, args.mean_pool, args.bert_model, args.do_lower_case)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_span_mask = torch.tensor([f.span_mask for f in train_features], dtype=torch.float)
        all_labels = torch.tensor([f.labels for f in train_features], dtype=torch.float)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_span_mask, all_labels)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        output_loss_file = os.path.join(args.output_dir, "loss")
        loss_fout = open(output_loss_file, 'w')
        model.train()
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) if i != 3 else t for i, t in enumerate(batch))
                input_ids, input_mask, segment_ids, span_mask, labels = batch
                loss = model(input_ids, args.bert_model, segment_ids, input_mask, span_mask, labels.half())
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                loss_fout.write("{}\n".format(loss.item()*args.gradient_accumulation_steps))
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    if global_step % 150 == 0 and global_step > 0:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin_{}".format(global_step))
                        torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(args.output_dir, "pytorch_model.bin_{}".format(epoch))
            torch.save(model_to_save.state_dict(), output_model_file)
            x = "pytorch_model.bin_{}".format(epoch)
            for mark in [True, False]:
                if mark:
                    eval_examples = processor.get_dev_examples(args.data_dir)
                else:
                    eval_examples = processor.get_test_examples(args.data_dir)
                eval_features = convert_examples_to_features(
                    eval_examples, label_list, args.max_seq_length, tokenizer_label, tokenizer, roberta_tokenizer, bert_tokenizer_cased, args.mean_pool, args.bert_model, args.do_lower_case)
                logger.info("***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
                all_span_mask = torch.tensor([f.span_mask for f in eval_features], dtype=torch.float)
                all_labels = torch.tensor([f.labels for f in eval_features], dtype=torch.float)
                eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_span_mask, all_labels)

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval()
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0
                pred = []
                true = []
                for input_ids, input_mask, segment_ids, span_mask, labels in eval_dataloader:
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    span_mask = span_mask.to(device)
                    labels = labels.to(device)

                    with torch.no_grad():
                        tmp_eval_loss = model(input_ids, args.bert_model, segment_ids, input_mask, span_mask, labels)
                        logits = model(input_ids, args.bert_model, segment_ids, input_mask, span_mask)

                    logits = logits.detach().cpu().numpy()
                    labels = labels.to('cpu').numpy()
                    tmp_eval_accuracy, tmp_pred, tmp_true = accuracy(logits, labels)
                    pred.extend(tmp_pred)
                    true.extend(tmp_true)

                    eval_loss += tmp_eval_loss.mean().item()
                    eval_accuracy += tmp_eval_accuracy

                    nb_eval_examples += input_ids.size(0)
                    nb_eval_steps += 1

                eval_loss = eval_loss / nb_eval_steps
                eval_accuracy = eval_accuracy / nb_eval_examples

                def f1(p, r):
                    if r == 0.:
                        return 0.
                    return 2 * p * r / float( p + r )
                def loose_macro(true, pred):
                    num_entities = len(true)
                    p = 0.
                    r = 0.
                    for true_labels, predicted_labels in zip(true, pred):
                        if len(predicted_labels) > 0:
                            p += len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
                        if len(true_labels):
                            r += len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
                    precision = p / num_entities
                    recall = r / num_entities
                    return precision, recall, f1( precision, recall)
                def loose_micro(true, pred):
                    num_predicted_labels = 0.
                    num_true_labels = 0.
                    num_correct_labels = 0.
                    for true_labels, predicted_labels in zip(true, pred):
                        num_predicted_labels += len(predicted_labels)
                        num_true_labels += len(true_labels)
                        num_correct_labels += len(set(predicted_labels).intersection(set(true_labels)))
                    if num_predicted_labels > 0:
                        precision = num_correct_labels / num_predicted_labels
                    else:
                        precision = 0.
                    recall = num_correct_labels / num_true_labels
                    return precision, recall, f1( precision, recall)

                if False:
                    result = {'eval_loss': eval_loss,
                            'eval_accuracy': eval_accuracy,
                            'macro': loose_macro(true, pred),
                            'micro': loose_micro(true, pred)
                            }
                else:
                    result = {'eval_loss': eval_loss,
                            'eval_accuracy': eval_accuracy,
                            'macro': loose_macro(true, pred),
                            'micro': loose_micro(true, pred)
                            }

                if mark:
                    output_eval_file = os.path.join(args.output_dir, "eval_results_{}.txt".format(x.split("_")[-1]))
                else:
                    output_eval_file = os.path.join(args.output_dir, "test_results_{}.txt".format(x.split("_")[-1]))
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results *****")
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))

    exit(0)

if __name__ == "__main__":
    main()
