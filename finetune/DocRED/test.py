import config
import numpy as np
import os
import time
import datetime
import json
from sklearn.metrics import average_precision_score
import sys
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="bert", required=True)
parser.add_argument("--model_name_or_path", type=str, default="bert-base-cased", required=True)
parser.add_argument("--prepro_data_dir", type=str, default="prepro_data", required=True)
parser.add_argument("--max_seq_length", default=512, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                            "longer than this will be truncated, and sequences shorter than this will be padded.")
parser.add_argument("--evaluate_during_training_epoch", default=5, type=int, help="evaluateing every X epochs")
parser.add_argument("--batch_size", default=32, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--learning_rate", default=4e-5, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=200, type=int, help="Total number of training epochs to perform.")
parser.add_argument('--seed', type=int, default=42,
                help="random seed for initialization")
parser.add_argument('--logging_steps', type=int, default=50,
                help="Log every X updates steps.")

parser.add_argument('--save_name', type = str, required=True)
parser.add_argument('--train_prefix', type = str, default = 'train')
parser.add_argument('--test_prefix', type = str, default = 'dev')
parser.add_argument("--input_threshold", default=-1, type=float, help="Prediction probability threshold")

args = parser.parse_args()

args = parser.parse_args()
con = config.Config(args)

con.testall(args.model_type, args.model_name_or_path, args.save_name, args.input_threshold)
