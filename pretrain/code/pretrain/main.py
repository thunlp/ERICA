import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import sys
import argparse
import sklearn.metrics
import matplotlib
import pdb
import numpy as np
import time
import random
import time
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from apex import amp
from tqdm import tqdm
from tqdm import trange
from sklearn import metrics
from torch.utils import data
from collections import Counter
from transformers import AdamW, get_linear_schedule_with_warmup
from dataset import *
from model import *

def log_loss(step_record, loss_record, args):
    if not os.path.exists("../../res"):
        os.mkdir("../../res")
    plt.plot(step_record, loss_record, lw=2)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('loss curve')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join("../../res", 'loss_curve_' + args.model + '_' + '.png'))
    plt.close()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def train(args, model, train_dataset):
    # total step
    step_tot = (len(train_dataset)  // args.gradient_accumulation_steps // args.batch_size_per_gpu // args.n_gpu) * args.max_epoch
    train_sampler = data.distributed.DistributedSampler(train_dataset) if args.local_rank != -1 else data.RandomSampler(train_dataset)
    params = {"batch_size": args.batch_size_per_gpu, "sampler": train_sampler, "collate_fn": train_dataset.get_train_batch}
    train_dataloader = data.DataLoader(train_dataset, **params)

    # optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=step_tot)

    # amp training
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # distributed training
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    print("Begin train...")
    print("We will train model in %d steps" % step_tot)
    global_step = 0
    loss_record = []
    step_record = []
    for i in range(args.max_epoch):
        for step, batch in enumerate(train_dataloader):
            if len(list(batch[1].keys())) == 0:
                continue
            batch = [{k: v.cuda() for k,v in b.items()} for b in batch]
            model.train()
            if args.doc_loss == 1:
                m_loss_d, r_loss_d = model(batch, doc_loss = 1, wiki_loss = 0)
                loss = m_loss_d + r_loss_d
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                m_loss_d = 0
                r_loss_d = 0
            if args.wiki_loss == 1:
                m_loss_w, r_loss_w = model(batch, doc_loss = 0, wiki_loss = 1)
                loss = m_loss_w + r_loss_w
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                m_loss_w = 0
                r_loss_w = 0

            if step % args.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [0, -1] and global_step % args.log_step == 0:
                    step_record.append(global_step)
                    loss_record.append(loss)

                if args.local_rank in [0, -1] and global_step % args.save_step == 0:
                    if not os.path.exists("../../ckpt"):
                        os.mkdir("../../ckpt")
                    if not os.path.exists("../../ckpt/"+args.save_dir):
                        os.mkdir("../../ckpt/"+args.save_dir)
                    if args.bert_model == 'bert':
                        ckpt = {
                            'bert-base': model.module.model.bert.state_dict(),
                        }
                    elif args.bert_model == 'roberta':
                        ckpt = {
                            'bert-base': model.module.model.roberta.state_dict(),
                        }
                    if global_step > 100:
                        torch.save(ckpt, os.path.join("../../ckpt/"+args.save_dir, "ckpt_of_step_"+str(global_step)))

                if not os.path.exists("***path_for_this_folder***"):
                    if args.local_rank in [0, -1]:
                        sys.stdout.write("step: %d, shcedule: %.3f, mlm_loss_d: %.6f, mlm_loss_w: %.6f, relation_loss_d: %.6f, relation_loss_w: %.6f \r" % (global_step, global_step/step_tot, m_loss_d, m_loss_w, r_loss_d, r_loss_w))
                        sys.stdout.flush()
                else:
                    if args.local_rank in [0, -1]:
                        print("step: %d, shcedule: %.3f, mlm_loss_d: %.6f, mlm_loss_w: %.6f, relation_loss_d: %.6f, relation_loss_w: %.6f \n" % (global_step, global_step/step_tot, m_loss_d, m_loss_w, r_loss_d, r_loss_w))

        if args.train_sample:
            print("sampling...")
            train_dataloader.dataset.__sample__()
            print("sampled")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="latentRE")
    parser.add_argument("--cuda", dest="cuda", type=str,
                        default="4", help="gpu id")

    parser.add_argument("--lr", dest="lr", type=float,
                        default=5e-5, help='learning rate')
    parser.add_argument("--batch_size_per_gpu", dest="batch_size_per_gpu", type=int,
                        default=32, help="batch size per gpu")
    parser.add_argument("--gradient_accumulation_steps", dest="gradient_accumulation_steps", type=int,
                        default=1, help="gradient accumulation steps")
    parser.add_argument("--max_epoch", dest="max_epoch", type=int,
                        default=3, help="max epoch number")

    parser.add_argument("--model", dest="model", type=str,
                        default="", help="{MTB, CP}")
    parser.add_argument("--train_sample",action="store_true",
                        help="dynamic sample or not")
    parser.add_argument("--max_length", dest="max_length", type=int,
                        default=512, help="max sentence length")
    parser.add_argument("--bag_size", dest="bag_size", type=int,
                        default=2, help="bag size")
    parser.add_argument("--temperature", dest="temperature", type=float,
                        default=0.05, help="temperature for NTXent loss")
    parser.add_argument("--hidden_size", dest="hidden_size", type=int,
                        default=768, help="hidden size for mlp")

    parser.add_argument("--weight_decay", dest="weight_decay", type=float,
                        default=1e-5, help="weight decay")
    parser.add_argument("--adam_epsilon", dest="adam_epsilon", type=float,
                        default=1e-8, help="adam epsilon")
    parser.add_argument("--warmup_steps", dest="warmup_steps", type=int,
                        default=500, help="warmup steps")
    parser.add_argument("--max_grad_norm", dest="max_grad_norm", type=float,
                        default=1, help="max grad norm")

    parser.add_argument("--save_step", dest="save_step", type=int,
                        default=10000, help="step to save")
    parser.add_argument("--save_dir", dest="save_dir", type=str,
                        default="", help="ckpt dir to save")

    parser.add_argument("--seed", dest="seed", type=int,
                        default=42, help="seed for network")

    parser.add_argument("--local_rank", dest="local_rank", type=int,
                        default=-1, help="local rank")

    parser.add_argument("--n_gpu", dest="n_gpu", type=int,
                        default=4, help="n_gpu")
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--log_step", type=int, default=5)
    parser.add_argument("--log_step_test", type=int, default=50)
    parser.add_argument("--curve_step", type=int, default=100)
    parser.add_argument("--neg_sample_num", type=int, default=64)
    parser.add_argument("--add_none", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--flow", type=int, default=0)
    parser.add_argument("--dataset_name", type=str, default="sampled_data/train_distant_debug.json")
    parser.add_argument("--doc_loss", type=float, default=1)
    parser.add_argument("--wiki_loss", type=float, default=1)
    parser.add_argument("--change_dataset", type=float, default=0)
    parser.add_argument("--start_end_token", type=float, default=0)
    parser.add_argument("--bert_model", type=str, default='bert')
    parser.add_argument("--pretraining_size", type=int, default=10)
    parser.add_argument("--ablation", type=float, default=0)
    parser.add_argument("--cased", type=float, default=0)

    args = parser.parse_args()
    # print args
    print(args)
    # set cuda
    if not os.path.exists("***path_to_this_folder***"):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
    args.device = device
    set_seed(args)

    # log train
    if args.local_rank in [0, -1]:
        if not os.path.exists("../../log"):
            os.mkdir("../../log")
        with open("../../log/pretrain_log", 'a+') as f:
            f.write(str(time.ctime())+"\n")
            f.write(str(args)+"\n")
            f.write("----------------------------------------------------------------------------\n")

    # Model and datase
    if args.model == "DOC":
        print('preparing data')
        if args.debug == 0:
            train_dataset = CP_R_Dataset("../../data/DOC", args)
        elif args.debug == 1:
            train_dataset = CP_R_Dataset("../../data/DOC", args)
        model = CP_R(args).to(args.device)
    else:
        raise Exception("No such model! Please make sure that `model` takes the value in {MTB, CP}")

    # Barrier to make sure all process train the model simultaneously.
    if args.local_rank != -1:
        torch.distributed.barrier()
    train(args, model, train_dataset)
