# coding: utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import os
import time
import datetime
import json
import sys
import sklearn.metrics
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import torch.nn.functional as F
from transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule, BertModel, RobertaModel, BertTokenizer, RobertaTokenizer)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import pickle
from REModel import REModel


MODEL_CLASSES = {
    'bert': BertModel,
    'roberta': RobertaModel,
}


IGNORE_INDEX = -100



class MyDataset():
    def __init__(self, prefix, data_path, h_t_limit, args):
        self.h_t_limit = h_t_limit

        self.data_path = data_path

        if args.ratio < 1 and prefix == 'train':
            self.train_file = json.load(open(os.path.join(self.data_path, prefix+'_' + str(args.ratio)+'.json')))

            self.data_train_bert_token = np.load(os.path.join(self.data_path, prefix+'_' + str(args.ratio)+'_bert_token.npy'))
            self.data_train_bert_mask = np.load(os.path.join(self.data_path, prefix+'_' + str(args.ratio)+'_bert_mask.npy'))
            self.data_train_bert_starts_ends = np.load(os.path.join(self.data_path, prefix+'_' + str(args.ratio)+'_bert_starts_ends.npy'))
        else:
            self.train_file = json.load(open(os.path.join(self.data_path, prefix+'.json')))

            self.data_train_bert_token = np.load(os.path.join(self.data_path, prefix+'_bert_token.npy'))
            self.data_train_bert_mask = np.load(os.path.join(self.data_path, prefix+'_bert_mask.npy'))
            self.data_train_bert_starts_ends = np.load(os.path.join(self.data_path, prefix+'_bert_starts_ends.npy'))


    def __getitem__(self, index):
        return self.train_file[index], self.data_train_bert_token[index],   \
                self.data_train_bert_mask[index], self.data_train_bert_starts_ends[index]

    def __len__(self):
        return self.data_train_bert_token.shape[0]

class Accuracy(object):
    def __init__(self):
        self.correct = 0
        self.total = 0
    def add(self, is_correct):
        self.total += 1
        if is_correct:
            self.correct += 1
    def get(self):
        if self.total == 0:
            return 0.0
        else:
            return float(self.correct) / self.total
    def clear(self):
        self.correct = 0
        self.total = 0

class Config(object):
    def __init__(self, args):
        self.acc_NA = Accuracy()
        self.acc_not_NA = Accuracy()
        self.acc_total = Accuracy()

        self.args = args

        self.max_seq_length = args.max_seq_length
        self.relation_num = 97

        self.max_epoch = args.num_train_epochs

        self.evaluate_during_training_epoch = args.evaluate_during_training_epoch

        self.log_period = args.logging_steps

        self.neg_multiple = 3 # The number of negative examples sampled is three times that of positive examples
        self.warmup_ratio = 0.1

        self.data_path = args.prepro_data_dir
        self.batch_size = args.batch_size
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.lr = args.learning_rate

        self.h_t_limit = 1800  # The maximum number of relation facts

        self.test_batch_size = self.batch_size * 2
        self.test_relation_limit = self.h_t_limit

        self.dis2idx = np.zeros((512), dtype='int64')
        self.dis2idx[1] = 1
        self.dis2idx[2:] = 2
        self.dis2idx[4:] = 3
        self.dis2idx[8:] = 4
        self.dis2idx[16:] = 5
        self.dis2idx[32:] = 6
        self.dis2idx[64:] = 7
        self.dis2idx[128:] = 8
        self.dis2idx[256:] = 9
        self.dis_size = 20

        self.train_prefix = args.train_prefix
        self.test_prefix = args.test_prefix

        self.checkpoint_dir = './checkpoint'
        self.fig_result_dir = './fig_result'


        if not os.path.exists("log"):
            os.mkdir("log")

        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")

        if not os.path.exists("fig_result"):
            os.mkdir("fig_result")

    def load_test_data(self):
        print("Reading testing data...")
        self.rel2id = json.load(open(os.path.join(self.data_path, 'rel2id.json')))
        self.id2rel = {v: k for k,v in self.rel2id.items()}

        prefix = self.test_prefix
        print (prefix)
        self.is_test = ('test' == prefix)
        self.test_file = json.load(open(os.path.join(self.data_path, prefix+'.json')))

        self.data_test_bert_token = np.load(os.path.join(self.data_path, prefix+'_bert_token.npy'))
        self.data_test_bert_mask = np.load(os.path.join(self.data_path, prefix+'_bert_mask.npy'))
        self.data_test_bert_starts_ends = np.load(os.path.join(self.data_path, prefix+'_bert_starts_ends.npy'))


        self.test_len = self.data_test_bert_token.shape[0]
        assert(self.test_len==len(self.test_file))

        print("Finish reading")

        self.test_batches = self.data_test_bert_token.shape[0] // self.test_batch_size
        if self.data_test_bert_token.shape[0] % self.test_batch_size != 0:
            self.test_batches += 1

        self.test_order = list(range(self.test_len))
        self.test_order.sort(key=lambda x: np.sum(self.data_test_bert_token[x] > 0), reverse=True)

    def get_test_batch(self):
        context_idxs = torch.LongTensor(self.test_batch_size, self.max_seq_length).cuda()
        h_mapping = torch.Tensor(self.test_batch_size, self.test_relation_limit, self.max_seq_length).cuda()
        t_mapping = torch.Tensor(self.test_batch_size, self.test_relation_limit, self.max_seq_length).cuda()

        relation_mask = torch.Tensor(self.test_batch_size, self.h_t_limit).cuda()
        ht_pair_pos = torch.LongTensor(self.test_batch_size, self.h_t_limit).cuda()

        context_masks = torch.LongTensor(self.test_batch_size, self.max_seq_length).cuda()

        for b in range(self.test_batches):
            start_id = b * self.test_batch_size
            cur_bsz = min(self.test_batch_size, self.test_len - start_id)
            cur_batch = list(self.test_order[start_id : start_id + cur_bsz])

            for mapping in [h_mapping, t_mapping, relation_mask]:
                mapping.zero_()

            ht_pair_pos.zero_()

            max_h_t_cnt = 1

            labels = []
            labels_multi = []

            L_vertex = []
            titles = []
            indexes = []

            evi_nums = []
            all_test_idxs = []
            all_test_idxs_0 = []
            all_test_idxs_1 = []
            all_test_idxs_2 = []
            all_test_idxs_multi = []
            for i, index in enumerate(cur_batch):
                context_idxs[i].copy_(torch.from_numpy(self.data_test_bert_token[index, :]))
                context_masks[i].copy_(torch.from_numpy(self.data_test_bert_mask[index, :]))

                idx2label = defaultdict(list)
                ins = self.test_file[index]
                starts_pos = self.data_test_bert_starts_ends[index, :, 0]
                ends_pos = self.data_test_bert_starts_ends[index, :, 1]

                for label in ins['labels']:
                    idx2label[(label['h'], label['t'])].append(label['r'])

                L = len(ins['vertexSet'])
                titles.append(ins['title'])

                j = 0
                test_idxs = []

                test_idxs_0 = []
                test_idxs_1 = []
                test_idxs_2 = []
                test_idxs_multi = []
                for h_idx in range(L):
                    for t_idx in range(L):
                        if h_idx != t_idx:
                            hlist = ins['vertexSet'][h_idx]
                            tlist = ins['vertexSet'][t_idx]

                            hlist = [ ( starts_pos[h['pos'][0]],  ends_pos[h['pos'][1]-1]  )  for h in hlist if ends_pos[h['pos'][1]-1]<511 ]
                            tlist = [ ( starts_pos[t['pos'][0]],  ends_pos[t['pos'][1]-1]  )  for t in tlist if ends_pos[t['pos'][1]-1]<511 ]
                            if len(hlist)==0 or len(tlist)==0:
                                continue

                            for h in hlist:
                                h_mapping[i, j, h[0]:h[1]] = 1.0 / len(hlist) / (h[1] - h[0])

                            for t in tlist:
                                t_mapping[i, j, t[0]:t[1]] = 1.0 / len(tlist) / (t[1] - t[0])

                            relation_mask[i, j] = 1

                            delta_dis = hlist[0][0] - tlist[0][0]
                            if delta_dis < 0:
                                ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                            else:
                                ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])

                            test_idxs.append((h_idx, t_idx))
                            j += 1


                max_h_t_cnt = max(max_h_t_cnt, j)
                label_set = {}

                for label in ins['labels']:
                    label_set[(label['h'], label['t'], label['r'])] = label['in_annotated_train']

                label_multi_set = {}
                for label in ins['labels']:
                    if len(label['evidence']) > 1:
                        test_idxs_2.append((label['h'], label['t']))
                        label_multi_set[(label['h'], label['t'])] = 2
                    elif len(label['evidence']) == 1:
                        test_idxs_1.append((label['h'], label['t']))
                        label_multi_set[(label['h'], label['t'])] = 1
                    elif len(label['evidence']) == 0:
                        test_idxs_0.append((label['h'], label['t']))
                        label_multi_set[(label['h'], label['t'])] = 0


                    hlist = [x['sent_id'] for x in ins['vertexSet'][label['h']]]
                    tlist = [x['sent_id'] for x in ins['vertexSet'][label['t']]]
                    flag = 0
                    for evi_idx in label['evidence']:
                        if evi_idx in hlist and evi_idx in tlist:
                            flag = 1
                    if flag == 0 and len(label['evidence']) > 1:
                        test_idxs_multi.append((label['h'], label['t']))

                labels.append(label_set)
                labels_multi.append(label_multi_set)

                L_vertex.append(L)
                indexes.append(index)
                all_test_idxs.append(test_idxs)

                all_test_idxs_0.append(test_idxs_0)
                all_test_idxs_1.append(test_idxs_1)
                all_test_idxs_2.append(test_idxs_2)
                all_test_idxs_multi.append(test_idxs_multi)


            max_c_len = self.max_seq_length

            yield {'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                #    'context_pos': context_pos[:cur_bsz, :max_c_len].contiguous(),
                   'h_mapping': h_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   't_mapping': t_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                   'labels': labels,
                   'label_multi': labels_multi,
                   'L_vertex': L_vertex,
                   'titles': titles,
                   'ht_pair_pos': ht_pair_pos[:cur_bsz, :max_h_t_cnt],
                   'indexes': indexes,
                   'context_masks': context_masks[:cur_bsz, :max_c_len].contiguous(),
                   'all_test_idxs': all_test_idxs,
                   'all_test_idxs_0': all_test_idxs_0,
                   'all_test_idxs_1': all_test_idxs_1,
                   'all_test_idxs_2': all_test_idxs_2,
                   'all_test_idxs_multi': all_test_idxs_multi,
                   }

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def get_train_batch(self, batch):
        batch_size = len(batch)
        max_length = self.max_seq_length
        h_t_limit = self.h_t_limit
        relation_num = self.relation_num
        context_idxs = torch.LongTensor(batch_size, max_length).zero_()
        h_mapping = torch.Tensor(batch_size, h_t_limit, max_length).zero_()
        t_mapping = torch.Tensor(batch_size, h_t_limit, max_length).zero_()
        relation_multi_label = torch.Tensor(batch_size, h_t_limit, relation_num).zero_()
        relation_mask = torch.Tensor(batch_size, h_t_limit).zero_()

        context_masks = torch.LongTensor(batch_size, self.max_seq_length).zero_()
        ht_pair_pos = torch.LongTensor(batch_size, h_t_limit).zero_()

        relation_label = torch.LongTensor(batch_size, h_t_limit).fill_(IGNORE_INDEX)

        for i, item in enumerate(batch):
            max_h_t_cnt = 1

            context_idxs[i].copy_(torch.from_numpy(item[1]))

            # xxx = context_idxs[i].tolist()
            # print(' '.join([tokenizer.decode(int(f)) for f in xxx]))
            # print(' '.join([idx2token[f] for f in xxx]))

            context_masks[i].copy_(torch.from_numpy(item[2]))
            starts_pos = item[3][:, 0]
            ends_pos = item[3][:, 1]

            ins = item[0]

            labels = ins['labels']
            idx2label = defaultdict(list)

            for label in labels:
                idx2label[(label['h'], label['t'])].append(label['r'])


            train_tripe = list(idx2label.keys())
            j = 0
            for (h_idx, t_idx) in train_tripe:
                if j == self.h_t_limit:
                    break
                hlist = ins['vertexSet'][h_idx]
                tlist = ins['vertexSet'][t_idx]

                # ddd = [xxx[int(starts_pos[int(h['pos'][0])]): int(ends_pos[int(h['pos'][1])-1])] for h in hlist]
                # for dddd in ddd:
                #     #print(' '.join([idx2token[x] for x in dddd]))
                #     print(' '.join([tokenizer.decode(int(x)) for x in dddd]))
                # print('\n')

                # ddd = [xxx[int(starts_pos[int(h['pos'][0])]): int(ends_pos[int(h['pos'][1])-1])] for h in tlist]
                # for dddd in ddd:
                #     print(' '.join([idx2token[x] for x in dddd]))
                # print('\n')

                hlist = [ ( starts_pos[h['pos'][0]],  ends_pos[h['pos'][1]-1]  )  for h in hlist if ends_pos[h['pos'][1]-1]<511 ]
                tlist = [ ( starts_pos[t['pos'][0]],  ends_pos[t['pos'][1]-1]  )  for t in tlist if ends_pos[t['pos'][1]-1]<511 ]
                if len(hlist)==0 or len(tlist)==0:
                    continue

                for h in hlist:
                    h_mapping[i, j, h[0]:h[1]] = 1.0 / len(hlist) / (h[1] - h[0])

                for t in tlist:
                    t_mapping[i, j, t[0]:t[1]] = 1.0 / len(tlist) / (t[1] - t[0])

                label = idx2label[(h_idx, t_idx)]

                delta_dis = hlist[0][0] - tlist[0][0]
                if delta_dis < 0:
                    ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                else:
                    ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])


                for r in label:
                    relation_multi_label[i, j, r] = 1

                relation_mask[i, j] = 1
                rt = np.random.randint(len(label))
                relation_label[i, j] = label[rt]

                j += 1

            lower_bound = min(len(ins['na_triple']), len(train_tripe) * self.neg_multiple)
            sel_idx = random.sample(list(range(len(ins['na_triple']))), lower_bound)
            sel_ins = [ins['na_triple'][s_i] for s_i in sel_idx]

            for (h_idx, t_idx) in sel_ins:
                if j == h_t_limit:
                    break
                hlist = ins['vertexSet'][h_idx]
                tlist = ins['vertexSet'][t_idx]

                # ddd = [xxx[int(starts_pos[int(h['pos'][0])]): int(ends_pos[int(h['pos'][1])-1])] for h in hlist]
                # for dddd in ddd:
                #     print(' '.join([idx2token[x] for x in dddd]))
                # print('\n')

                # ddd = [xxx[int(starts_pos[int(h['pos'][0])]): int(ends_pos[int(h['pos'][1])-1])] for h in tlist]
                # for dddd in ddd:
                #     print(' '.join([idx2token[x] for x in dddd]))
                # print('\n')

                hlist = [ ( starts_pos[h['pos'][0]],  ends_pos[h['pos'][1]-1]  )  for h in hlist if ends_pos[h['pos'][1]-1]<511 ]
                tlist = [ ( starts_pos[t['pos'][0]],  ends_pos[t['pos'][1]-1]  )  for t in tlist if ends_pos[t['pos'][1]-1]<511 ]
                if len(hlist)==0 or len(tlist)==0:
                    continue

                for h in hlist:
                    h_mapping[i, j, h[0]:h[1]] = 1.0 / len(hlist) / (h[1] - h[0])

                for t in tlist:
                    t_mapping[i, j, t[0]:t[1]] = 1.0 / len(tlist) / (t[1] - t[0])


                delta_dis = hlist[0][0] - tlist[0][0]

                relation_multi_label[i, j, 0] = 1
                relation_label[i, j] = 0
                relation_mask[i, j] = 1

                if delta_dis < 0:
                    ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                else:
                    ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])
                j += 1

            max_h_t_cnt = max(max_h_t_cnt, len(train_tripe) + lower_bound)

        return {'context_idxs': context_idxs,
                'h_mapping': h_mapping[:, :max_h_t_cnt, :].contiguous(),
                't_mapping': t_mapping[:, :max_h_t_cnt, :].contiguous(),
                'relation_label': relation_label[:, :max_h_t_cnt].contiguous(),
                'relation_multi_label': relation_multi_label[:, :max_h_t_cnt].contiguous(),
                'relation_mask': relation_mask[:, :max_h_t_cnt].contiguous(),
                'ht_pair_pos': ht_pair_pos[:, :max_h_t_cnt].contiguous(),
                'context_masks': context_masks,
                }

    def train(self, model_type, model_name_or_path, save_name, args):
        self.load_test_data()

        train_dataset = MyDataset(self.train_prefix, self.data_path, self.h_t_limit, args)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.batch_size, collate_fn=self.get_train_batch, num_workers=2)
        bert_model = MODEL_CLASSES[model_type].from_pretrained(model_name_or_path)

        if args.ckpt != 'None':
            if os.path.exists('***path_to_your_trained_ckpt***'):
                load_path = '***path_to_your_trained_ckpt***' + args.ckpt
            else:
                load_path = '***path_to_your_trained_ckpt***' + args.ckpt

            ckpt = torch.load(load_path)
            bert_model.load_state_dict(ckpt["bert-base"])

        ori_model = REModel(config = self, bert_model=bert_model)
        ori_model.cuda()

        model = nn.DataParallel(ori_model)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, eps=self.args.adam_epsilon)
        tot_step = int( (len(train_dataset) // self.batch_size+1) / self.gradient_accumulation_steps * self.max_epoch)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=int(self.warmup_ratio*tot_step), t_total=tot_step)

        save_step = int( (len(train_dataset) // self.batch_size+1) / self.gradient_accumulation_steps  * self.evaluate_during_training_epoch)
        print ("tot_step:", tot_step, "save_step:", save_step, self.lr)

        BCE = nn.BCEWithLogitsLoss(reduction='none')

        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)


        best_all_f1_d = 0.0
        best_result = None
        best_all_auc_d = 0
        best_all_auc_t = 0
        best_all_epoch = 0

        model.train()

        global_step = 0
        total_loss = 0
        start_time = time.time()

        def logging(s, print_=True, log_=True):
            if print_:
                print(s)
            if log_:
                with open(os.path.join(os.path.join("log", save_name)), 'a+') as f_log:
                    f_log.write(s + '\n')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim(0.3, 1.0)
        plt.xlim(0.0, 0.4)
        plt.title('Precision-Recall')
        plt.grid(True)

        step = 0
        for epoch in range(self.max_epoch):
            self.acc_NA.clear()
            self.acc_not_NA.clear()
            self.acc_total.clear()

            for batch in train_dataloader:
                data = {k: v.cuda() for k,v in batch.items()}

                context_idxs = data['context_idxs']
                h_mapping = data['h_mapping']
                t_mapping = data['t_mapping']
                relation_label = data['relation_label']
                relation_multi_label = data['relation_multi_label']
                relation_mask = data['relation_mask']

                ht_pair_pos = data['ht_pair_pos']
                context_masks = data['context_masks']

                if torch.sum(relation_mask)==0:
                    print ('zero input')
                    continue

                dis_h_2_t = ht_pair_pos+10
                dis_t_2_h = -ht_pair_pos+10

                predict_re = model(context_idxs, h_mapping, t_mapping, dis_h_2_t, dis_t_2_h, context_masks)

                pred_loss = BCE(predict_re, relation_multi_label)*relation_mask.unsqueeze(2)

                loss = torch.sum(pred_loss) /  (self.relation_num * torch.sum(relation_mask))
                if torch.isnan(loss):
                    pickle.dump(data, open("crash_data.pkl","wb"))
                    path = os.path.join(self.checkpoint_dir, model_name+"_crash")
                    torch.save(ori_model.state_dict(), path)


                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                output = torch.argmax(predict_re, dim=-1)
                output = output.data.cpu().numpy()

                loss.backward()

                relation_label = relation_label.data.cpu().numpy()

                for i in range(output.shape[0]):
                    for j in range(output.shape[1]):
                        label = relation_label[i][j]
                        if label < 0:
                            break

                        if label == 0:
                            self.acc_NA.add(output[i][j] == label)
                        else:
                            self.acc_not_NA.add(output[i][j] == label)

                        self.acc_total.add(output[i][j] == label)

                total_loss += loss.item()

                if (step + 1) % self.gradient_accumulation_steps == 0:

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if global_step % self.log_period == 0 :
                        cur_loss = total_loss / self.log_period
                        elapsed = time.time() - start_time
                        logging('| epoch {:2d} | step {:4d} |  ms/b {:5.2f} | train loss {:.8f} | NA acc: {:4.2f} | not NA acc: {:4.2f}  | tot acc: {:4.2f} '.format(
                            epoch, global_step, elapsed * 1000 / self.log_period, cur_loss, self.acc_NA.get(), self.acc_not_NA.get(), self.acc_total.get()))
                        total_loss = 0
                        start_time = time.time()

                    if global_step % save_step == 0:
                        logging('-' * 89)
                        eval_start_time = time.time()
                        if epoch > 0.3 * self.max_epoch:
                            model.eval()
                            logging('dev set evaluation')
                            self.test_prefix = 'dev'
                            self.load_test_data()
                            all_f1_d, test_f1_d, ign_f1_d, f1_d, auc_d, pr_x_d, pr_y_d, input_theta_dev = self.test(model, save_name)

                            logging('test set evaluation')
                            self.test_prefix = 'test'
                            self.load_test_data()
                            all_f1_t, test_f1_t, ign_f1_t, f1_t, auc_t, pr_x_t, pr_y_t, input_theta_test = self.test(model, save_name, False, input_theta_dev)

                            if all_f1_d > best_all_f1_d:
                                best_all_f1_d = all_f1_d
                                best_result = {'dev': [all_f1_d, test_f1_d, ign_f1_d, f1_d, auc_d, input_theta_dev], 'test':
                                        [all_f1_t, test_f1_t, ign_f1_t, f1_t, auc_t, input_theta_test]}
                                best_all_epoch = epoch

                        model.train()
                        logging('| epoch {:3d} | time: {:5.2f}s'.format(epoch, time.time() - eval_start_time))
                        logging('-' * 89)

                step += 1

        logging('-' * 89)
        eval_start_time = time.time()
        model.eval()
        logging('dev set evaluation')
        self.test_prefix = 'dev'
        self.load_test_data()
        all_f1_d, test_f1_d, ign_f1_d, f1_d, auc_d, pr_x_d, pr_y_d, input_theta_dev = self.test(model, save_name)

        logging('test set evaluation')
        self.test_prefix = 'test'
        self.load_test_data()
        all_f1_t, test_f1_t, ign_f1_t, f1_t, auc_t, pr_x_t, pr_y_t, input_theta_test = self.test(model, save_name, False, input_theta_dev)

        # model.train()
        # logging('| epoch {:3d} | time: {:5.2f}s'.format(epoch, time.time() - eval_start_time))
        # logging('-' * 89)

        if all_f1_d > best_all_f1_d:
            best_all_f1_d = all_f1_d
            best_result = {'dev': [all_f1_d, test_f1_d, ign_f1_d, f1_d, auc_d, input_theta_dev], 'test':
                    [all_f1_t, test_f1_t, ign_f1_t, f1_t, auc_t, input_theta_test]}
            best_all_epoch = epoch

        print("Finish training")
        print("Best epoch = %d" % (best_all_epoch))
        print(best_result)

    def test(self, model, save_name, output=False, input_theta=-1):
        data_idx = 0
        eval_start_time = time.time()
        # test_result_ignore = []
        total_recall_ignore = 0

        test_result = []
        total_recall = 0
        top1_acc = have_label = 0
        top1_acc_0 = 0
        top1_acc_1 = 0
        top1_acc_2 = 0
        top1_acc_multi = 0

        predicted_as_zero = 0
        total_ins_num = 0
        total_ins_num_0 = 0
        total_ins_num_1 = 0
        total_ins_num_2 = 0
        total_ins_num_multi = 0

        def logging(s, print_=True, log_=True):
            if print_:
                print(s)
            if log_:
                with open(os.path.join(os.path.join("log", save_name)), 'a+') as f_log:
                    f_log.write(s + '\n')

        for data in self.get_test_batch():
            with torch.no_grad():
                context_idxs = data['context_idxs']
                h_mapping = data['h_mapping']
                t_mapping = data['t_mapping']
                labels = data['labels']
                labels_multi = data['label_multi']
                L_vertex = data['L_vertex']
                ht_pair_pos = data['ht_pair_pos']
                context_masks = data['context_masks']
                all_test_idxs = data['all_test_idxs']

                all_test_idxs_0 = data['all_test_idxs_0']
                all_test_idxs_1 = data['all_test_idxs_1']
                all_test_idxs_2 = data['all_test_idxs_2']
                all_test_idxs_multi = data['all_test_idxs_multi']

                titles = data['titles']
                indexes = data['indexes']

                dis_h_2_t = ht_pair_pos+10
                dis_t_2_h = -ht_pair_pos+10

                predict_re = model(context_idxs, h_mapping, t_mapping, dis_h_2_t, dis_t_2_h, context_masks)

                predict_re = torch.sigmoid(predict_re)

            predict_re = predict_re.data.cpu().numpy()
            for i in range(len(labels)):
                label = labels[i]
                label_multi = labels_multi[i]
                index = indexes[i]
                all_test_idx_0 = all_test_idxs_0[i]
                all_test_idx_1 = all_test_idxs_1[i]
                all_test_idx_2 = all_test_idxs_2[i]
                all_test_idx_multi = all_test_idxs_multi[i]

                total_recall += len(label)
                for l in label.values():
                    if not l:
                        total_recall_ignore += 1

                L = L_vertex[i]
                test_idxs = all_test_idxs[i]
                j = 0

                for (h_idx, t_idx) in test_idxs:
                    r = np.argmax(predict_re[i, j])
                    predicted_as_zero += (r==0)
                    total_ins_num += 1

                    for r in range(1, self.relation_num):
                        intrain = False
                        multi = -1
                        if (h_idx, t_idx, r) in label:
                            if label[(h_idx, t_idx, r)]==True:
                                intrain = True
                            multi = label_multi[(h_idx, t_idx)]
                        test_result.append( ((h_idx, t_idx, r) in label, float(predict_re[i,j,r]), (intrain, multi),  titles[i], self.id2rel[r], index, h_idx, t_idx, r) )
                    j += 1

            data_idx += 1

            if data_idx % self.log_period == 0:
                print('| step {:3d} | time: {:5.2f}'.format(data_idx // self.log_period, (time.time() - eval_start_time)))
                eval_start_time = time.time()

        test_result.sort(key = lambda x: x[1], reverse=True)

        pr_x = []
        pr_y = []
        correct = 0
        w = 0

        if total_recall == 0:
            total_recall = 1  # for test

        for i, item in enumerate(test_result):
            correct += item[0]
            pr_y.append(float(correct) / (i + 1))
            pr_x.append(float(correct) / total_recall)
            if item[1] > input_theta:
                w = i


        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        f1 = f1_arr.max()
        f1_pos = f1_arr.argmax()
        all_f1 = f1
        theta = test_result[f1_pos][1]

        if input_theta==-1:
            w = f1_pos
            input_theta = theta

        auc = sklearn.metrics.auc(x = pr_x, y = pr_y)
        if not self.is_test:
            logging('ALL  : Theta {:3.4f} | F1 {:3.4f} | AUC {:3.4f}'.format(theta, f1, auc))
        else:
            logging('ma_f1 {:3.4f} | input_theta {:3.4f} test_result F1 {:3.4f} | AUC {:3.4f}'.format(f1, input_theta, f1_arr[w], auc))
        test_f1 = f1_arr[w]

        if output:
            output = [{'index': x[-4], 'h_idx': x[-3], 't_idx': x[-2], 'r_idx': x[-1], 'r': x[-5], 'title': x[-6]} for x in test_result[:w+1]]
            json.dump(output, open(save_name + "_" + self.test_prefix + "_index.json", "w"))
            print ('finish output')

        pr_x = []
        pr_y = []
        correct = correct_in_train = 0
        w = 0
        for i, item in enumerate(test_result):
            correct += item[0]
            if item[0] & item[2][0]:
                correct_in_train += 1
            if correct_in_train==correct:
                p = 0
            else:
                p = float(correct - correct_in_train) / (i + 1 - correct_in_train)
            pr_y.append(p)
            pr_x.append(float(correct) / total_recall)
            if item[1] > input_theta:
                w = i

        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        ign_f1 = f1_arr.max()

        auc = sklearn.metrics.auc(x = pr_x, y = pr_y)

        logging('Ignore ma_f1 {:3.4f} | input_theta {:3.4f} test_result F1 {:3.4f} | AUC {:3.4f}'.format(ign_f1, input_theta, f1_arr[w], auc))

        return all_f1, test_f1, ign_f1, f1_arr[w], auc, pr_x, pr_y, input_theta


    def testall(self, model_type, model_name_or_path, save_name, input_theta):
        self.load_test_data()
        bert_model = MODEL_CLASSES[model_type].from_pretrained(model_name_or_path)
        model = REModel(config = self, bert_model=bert_model)

        model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, save_name)))
        model.cuda()
        model = nn.DataParallel(model)
        model.eval()

        self.test(model, save_name, True, input_theta)
