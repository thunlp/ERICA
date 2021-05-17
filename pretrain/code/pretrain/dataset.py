import json
import random
import os
import sys
sys.path.append("..")
import pdb
import re
import pdb
import math
import torch
import numpy as np
from collections import Counter
from torch.utils import data
from utils import EntityMarker
from collections import defaultdict
import re

class CP_R_Dataset(data.Dataset):
    def __init__(self, path, args):
        self.path = path
        self.args = args
        if args.bert_model == 'bert':
            if args.cased == 0:
                self.P_info = json.load(open('../../data/DOC/P_info_tokenized.json', 'r'))
            elif args.cased == 1:
                self.P_info = json.load(open('../../data/DOC/P_info_tokenized_bert_base_cased.json', 'r'))
        elif args.bert_model == 'roberta':
            self.P_info = json.load(open('../../data/DOC/P_info_tokenized_roberta.json', 'r'))

        self.rel2id = json.load(open(os.path.join(path, "relation2idx.json")))
        self.h_t_limit = 1000
        self.neg_multiple = 32
        self.relation_num = len(list(self.rel2id.keys()))
        self.max_length = args.max_length

        self.entityMarker = EntityMarker(args)
        if args.bert_model == 'bert':
            self.idx2token = {v: k for k,v in self.entityMarker.tokenizer.vocab.items()}

        self.type2mask = ['[unused' + str(x) + ']' for x in range(1, 101)]
        self.start_token = ['[unused' + str(x) + ']' for x in range(101, 201)]
        self.end_token = ['[unused' + str(x) + ']' for x in range(201, 301)]
        self.alpha = self.args.alpha
        self.alpha_grad = (1.0 - self.alpha) / self.args.max_epoch
        self.epoch = 0

        self.__sample__()

    def get_data(self, ori_data, data_type):
        data = []
        for i in range(len(ori_data)):
            vertexSet = ori_data[i]['vertexSet']

            mask_shuffled = list(range(100))
            random.shuffle(mask_shuffled)
            start_end_shuffled = list(range(100))
            random.shuffle(start_end_shuffled)

            mask_idx = 0
            start_end_idx = 0
            sent_id_to_vertex = {}

            if data_type == 'wiki_data':
                time = 0
                while(True):
                    if time > 10000:
                        print('!!!')
                    time += 1
                    hop_rel = random.choice(ori_data[i]['labels'])

                    if hop_rel['r'] in self.P_info:
                        break

                query_h = hop_rel['h']
                query_t = hop_rel['t']
                query_r = hop_rel['r']

                query = []
                if self.args.bert_model == 'bert':
                    query += self.P_info[query_r]
                    query += list(map(self.entityMarker.tokenize, ori_data[i]['vertexSet'][query_h][0]['name'].split(' ')))
                    query += [[self.entityMarker.tokenizer.sep_token]]
                elif self.args.bert_model == 'roberta':
                    fflag = 0
                    for ddd_word in self.P_info[query_r]:
                        query += ddd_word
                    #query += ori_data[i]['vertexSet'][query_h][0]['name'].split(' ')
                    sent_id_q = ori_data[i]['vertexSet'][query_h][0]['sent_id']
                    pos_q = ori_data[i]['vertexSet'][query_h][0]['pos']
                    sent_q = ori_data[i]['sents'][sent_id_q]
                    query += sent_q[pos_q[0]: pos_q[1]]

                    query += [self.entityMarker.tokenizer.sep_token]

                if self.args.bert_model == 'bert':
                    ori_data[i]['tokenized_sents'][0][0: 0] = iter(query)
                elif self.args.bert_model == 'roberta':
                    ori_data[i]['sents'][0][0: 0] = iter(query)
                for x in range(len(ori_data[i]['vertexSet'])):
                    for y in range(len(ori_data[i]['vertexSet'][x])):
                        if ori_data[i]['vertexSet'][x][y]['sent_id'] == 0:
                            for z in range(len(ori_data[i]['vertexSet'][x][y]['pos'])):
                                ori_data[i]['vertexSet'][x][y]['pos'][z] += len(query)

                ori_data[i]['label_hop'] = hop_rel

                for jj in range(len(vertexSet)):
                    for k in range(len(vertexSet[jj])):
                        sent_id = int(vertexSet[jj][k]['sent_id'])
                        if sent_id not in sent_id_to_vertex:
                            sent_id_to_vertex[sent_id] = []
                        sent_id_to_vertex[sent_id].append([jj, k])

                for j in range(len(vertexSet)):
                    if j == query_h:
                        continue
                    if random.random() > self.alpha:
                        for k in range(len(vertexSet[j])):
                            sent_id = int(vertexSet[j][k]['sent_id'])
                            sent = ori_data[i]['tokenized_sents'][sent_id]
                            pos1 = vertexSet[j][k]['pos'][0]
                            pos2 = vertexSet[j][k]['pos'][1]
                            ori_data[i]['tokenized_sents'][sent_id] = sent[: pos1] + [[self.type2mask[mask_shuffled[mask_idx]]]] + sent[pos2: ]
                            for x,y in sent_id_to_vertex[sent_id]:
                                if x != j or y != k:
                                    if vertexSet[x][y]['pos'][0] >= pos2:
                                        vertexSet[x][y]['pos'][0] -= pos2 - pos1 - 1
                                        vertexSet[x][y]['pos'][1] -= pos2 - pos1 - 1
                                    elif vertexSet[x][y]['pos'][0] == pos1 and vertexSet[x][y]['pos'][1] == pos2:
                                        vertexSet[x][y]['pos'][1] = pos1 + 1
                            vertexSet[j][k]['pos'][1] = pos1 + 1
                        mask_idx += 1

            elif data_type == 'doc_data':
                for jj in range(len(vertexSet)):
                    for k in range(len(vertexSet[jj])):
                        sent_id = int(vertexSet[jj][k]['sent_id'])
                        if sent_id not in sent_id_to_vertex:
                            sent_id_to_vertex[sent_id] = []
                        sent_id_to_vertex[sent_id].append([jj, k])
            else:
                assert False

            if self.args.start_end_token == 1:
                for j in range(len(vertexSet)):
                    if self.args.bert_model == 'bert':
                        start_token = self.start_token[start_end_shuffled[start_end_idx]]
                        end_token = self.end_token[start_end_shuffled[start_end_idx]]
                    elif self.args.bert_model == 'roberta':
                        start_token = self.entityMarker.tokenizer.bos_token
                        end_token = self.entityMarker.tokenizer.eos_token
                    for k in range(len(vertexSet[j])):
                        sent_id = int(vertexSet[j][k]['sent_id'])
                        if self.args.bert_model == 'bert':
                            sent = ori_data[i]['tokenized_sents'][sent_id]
                        elif self.args.bert_model == 'roberta':
                            sent = ori_data[i]['sents'][sent_id]
                        pos1 = vertexSet[j][k]['pos'][0]
                        pos2 = vertexSet[j][k]['pos'][1]
                        if self.args.bert_model == 'bert':
                            ori_data[i]['tokenized_sents'][sent_id] = sent[: pos1] + [[start_token]] + sent[pos1: pos2] + [[end_token]] + sent[pos2: ]
                        elif self.args.bert_model == 'roberta':
                            ori_data[i]['sents'][sent_id] = sent[: pos1] + [start_token] + sent[pos1: pos2] + [end_token] + sent[pos2: ]
                        for x,y in sent_id_to_vertex[sent_id]:
                            if x != j or y != k:
                                if vertexSet[x][y]['pos'][0] >= pos2:
                                    vertexSet[x][y]['pos'][0] += 2
                                    vertexSet[x][y]['pos'][1] += 2
                                elif vertexSet[x][y]['pos'][0] == pos1 and vertexSet[x][y]['pos'][1] == pos2:
                                    vertexSet[x][y]['pos'][0] += 1
                                    vertexSet[x][y]['pos'][1] += 1
                        vertexSet[j][k]['pos'][0] += 1
                        vertexSet[j][k]['pos'][1] += 1
                    start_end_idx += 1

            Ls = [0]
            L = 0
            if self.args.bert_model == 'bert':
                for x in ori_data[i]['tokenized_sents']:
                    L += len(x)
                    Ls.append(L)
            elif self.args.bert_model == 'roberta':
                for x in ori_data[i]['sents']:
                    L += len(x)
                    Ls.append(L)
            for j in range(len(vertexSet)):
                for k in range(len(vertexSet[j])):
                    vertexSet[j][k]['sent_id'] = int(vertexSet[j][k]['sent_id'])
                    sent_id = vertexSet[j][k]['sent_id']
                    dl = Ls[sent_id]
                    pos1 = vertexSet[j][k]['pos'][0]
                    pos2 = vertexSet[j][k]['pos'][1]
                    vertexSet[j][k]['pos'] = (pos1+dl, pos2+dl)

            ori_data[i]['vertexSet'] = vertexSet

        return ori_data

    def __sample__(self):
        if self.args.flow == 1:
            self.alpha += self.alpha_grad
            print('alpha: ' + str(self.alpha))
        if self.args.debug == 0:
            if not self.args.change_dataset:
                train_data = json.load(open(os.path.join(self.path, self.args.dataset_name), 'r'))
            else:
                if self.args.bert_model == 'bert':
                    if self.args.dataset_name == 'sampled_data_docred':
                        train_data = json.load(open(os.path.join(self.path, 'sampled_data_docred/train_distant_' + str(self.epoch % 2) + '.json'), 'r'))
                        self.epoch += 1
                    elif self.args.dataset_name == 'sampled_data_docred_no_remove':
                        train_data = json.load(open(os.path.join(self.path, 'sampled_data_docred_no_remove/train_distant_' + str(self.epoch % 2) + '.json'), 'r'))
                        self.epoch += 1
                    elif self.args.dataset_name == 'sampled_data_bert_base_cased':
                        train_data = json.load(open(os.path.join(self.path, 'sampled_data_bert_base_cased/train_distant_' + str(self.epoch % 2) + '.json'), 'r'))
                        self.epoch += 1
                    else:
                        train_data = json.load(open(os.path.join(self.path, 'sampled_data/train_distant_' + str(self.epoch % self.args.pretraining_size) + '.json'), 'r'))
                        self.epoch += 1
                elif self.args.bert_model == 'roberta':
                    if self.args.dataset_name == 'sampled_data_docred_roberta':
                        train_data = json.load(open(os.path.join(self.path, 'sampled_data_docred_roberta/train_distant_' + str(self.epoch % 2) + '.json'), 'r'))
                        self.epoch += 1
                    elif self.args.dataset_name == 'sampled_data_docred_no_remove_roberta':
                        train_data = json.load(open(os.path.join(self.path, 'sampled_data_docred_no_remove_roberta/train_distant_' + str(self.epoch % 2) + '.json'), 'r'))
                        self.epoch += 1
                    else:
                        train_data = json.load(open(os.path.join(self.path, 'sampled_data_roberta/train_distant_' + str(self.epoch % self.args.pretraining_size) + '.json'), 'r'))
                        self.epoch += 1
                else:
                    assert False
        elif self.args.debug == 1:
            if self.args.bert_model == 'bert':
                if self.args.cased == 0:
                    train_data = json.load(open(os.path.join(self.path, "train_distant_debug.json"), 'r'))
                elif self.args.cased == 1:
                    train_data = json.load(open(os.path.join(self.path, "train_distant_cased_debug.json"), 'r'))
            elif self.args.bert_model == 'roberta':
                train_data = json.load(open(os.path.join(self.path, "train_distant_roberta_debug.json"), 'r'))

        random.shuffle(train_data)
        self.half_data_len = int(0.5 * len(train_data))

        self.doc_data = self.get_data(train_data[: self.half_data_len], 'doc_data')
        self.wiki_data = self.get_data(train_data[self.half_data_len: 2 * self.half_data_len], 'wiki_data')

        self.tokens = np.zeros((len(self.doc_data) + len(self.wiki_data), self.args.max_length), dtype=int)
        self.mask = np.zeros((len(self.doc_data) + len(self.wiki_data), self.args.max_length), dtype=int)
        self.bert_starts_ends = np.ones((len(self.doc_data) + len(self.wiki_data), self.args.max_length, 2), dtype = np.int64) * (self.args.max_length - 1)

        self.numberize(self.doc_data, 0)
        self.numberize(self.wiki_data, self.half_data_len)

        self.pos_pair = []
        scope = list(range(self.half_data_len))
        random.shuffle(scope)
        self.pos_pair = scope
        print("Postive pair's number is %d" % (len(self.pos_pair)))

    def numberize(self, data, offset):
        self.bad_instance_id = []
        wired_example_num = 0
        for i in range(len(data)):
            item = data[i]
            if self.args.bert_model == 'bert':
                subwords = []
                for sent in item['tokenized_sents']:
                    subwords += sent

                subword_lengths = list(map(len, subwords))
                flatten_subwords = [x for x_list in subwords for x in x_list ]

                tokens = [self.entityMarker.tokenizer.cls_token] + flatten_subwords[: self.args.max_length - 2] + [self.entityMarker.tokenizer.sep_token]
                token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
                token_start_idxs[token_start_idxs >= self.args.max_length-1] = self.args.max_length - 1
                token_end_idxs = 1 + np.cumsum(subword_lengths)
                token_end_idxs[token_end_idxs >= self.args.max_length-1] = self.args.max_length - 1
                token_start_idxs = token_start_idxs.tolist()
                token_end_idxs = token_end_idxs.tolist()
                tokens = self.entityMarker.tokenizer.convert_tokens_to_ids(tokens)
                pad_len = self.args.max_length - len(tokens)
                mask = [1] * len(tokens) + [0] * pad_len
                tokens = tokens + [0] * pad_len
                self.tokens[i + offset] = tokens
                self.mask[i + offset] = mask
                self.bert_starts_ends[i + offset, :len(subword_lengths), 0] = token_start_idxs[: self.args.max_length]
                self.bert_starts_ends[i + offset, :len(subword_lengths), 1] = token_end_idxs[: self.args.max_length]
            elif self.args.bert_model == 'roberta':
                words = []
                for sent in item['sents']:
                    words += sent
                idxs = []
                text = ""
                for word in words:
                    if len(text) > 0:
                        text = text  + " "
                    idxs.append(len(text))
                    text += word

                subwords = self.entityMarker.tokenizer.tokenize(text)

                char2subwords = []
                L = 0
                sub_idx = 0
                L_subwords = len(subwords)
                while sub_idx < L_subwords:
                    subword_list = []
                    prev_sub_idx = sub_idx
                    while sub_idx < L_subwords:
                        subword_list.append(subwords[sub_idx])
                        sub_idx += 1
                        subword = self.entityMarker.tokenizer.convert_tokens_to_string(subword_list)
                        sub_l = len(subword)
                        if subword == '</s>':
                            L += 1
                        if text[L:L+sub_l] == subword:
                            break
                    assert(text[L:L+sub_l]==subword)
                    if subword == '</s>':
                        char2subwords.extend([prev_sub_idx] * (sub_l + 1))
                    else:
                        char2subwords.extend([prev_sub_idx] * sub_l)

                    L += len(subword)

                if len(text) != len(char2subwords):
                    wired_example_num += 1
                    self.bad_instance_id.append(i + offset)
                    continue
                    # text = text[:len(char2subwords)]

                assert(len(text)==len(char2subwords))
                tokens = [ self.entityMarker.tokenizer.cls_token ] + subwords[: 512 - 2] + [ self.entityMarker.tokenizer.sep_token ]

                L_ori = len(tokens)
                tokens = self.entityMarker.tokenizer.convert_tokens_to_ids(tokens)

                pad_len = 512 - len(tokens)
                mask = [1] * len(tokens) + [0] * pad_len
                tokens = tokens + [1] * pad_len

                self.tokens[i + offset] = tokens
                self.mask[i + offset] = mask

                for j in range(len(words)):
                    idx = char2subwords[idxs[j]] + 1
                    idx = min(idx, 512-1)

                    x = idxs[j] + len(words[j])
                    if x == len(char2subwords):
                        idx2 = L_ori
                    else:
                        idx2 = char2subwords[x] + 1
                        idx2 = min(idx2, 512-1)
                    self.bert_starts_ends[i + offset, j, 0] = idx
                    self.bert_starts_ends[i + offset, j, 1] = idx2

        print('wired_example_num: ' + str(wired_example_num))

    def __len__(self):
        return len(self.pos_pair)

    def __getitem__(self, index):
        while index in self.bad_instance_id or index + self.half_data_len in self.bad_instance_id:
            print('reselecting another instance')
            index = random.choice(list(range(500)))
        bag_idx = self.pos_pair[index]

        ids = self.tokens[bag_idx]
        mask = self.mask[bag_idx]
        bert_starts_ends = self.bert_starts_ends[bag_idx]
        item = self.doc_data[bag_idx]

        ids_w = self.tokens[bag_idx + self.half_data_len]
        mask_w = self.mask[bag_idx + self.half_data_len]
        bert_starts_ends_w = self.bert_starts_ends[bag_idx + self.half_data_len]
        item_w = self.wiki_data[bag_idx]

        return (ids, mask, bert_starts_ends, item), (ids_w, mask_w, bert_starts_ends_w, item_w)

    def get_doc_batch(self, batch):
        batch_size = len(batch)

        max_length = self.max_length
        h_t_limit = self.h_t_limit
        context_idxs = torch.LongTensor(batch_size, max_length).zero_()
        h_mapping = torch.Tensor(batch_size, h_t_limit, max_length).zero_()
        t_mapping = torch.Tensor(batch_size, h_t_limit, max_length).zero_()
        mlm_mask = torch.Tensor(batch_size, max_length).zero_()
        context_masks = torch.LongTensor(batch_size, max_length).zero_()

        rel_list = {}
        rel_list_none = []
        max_h_t_cnt = -1
        idx2pair = {}
        for i in range(len(batch)):
            item = batch[i]
            context_idxs[i].copy_(torch.from_numpy(item[0]))
            context_masks[i].copy_(torch.from_numpy(item[1]))

            item = batch[i][3]
            starts_pos = batch[i][2][:, 0]
            ends_pos = batch[i][2][:, 1]
            labels = item['labels']

            idx2label = defaultdict(list)

            for label in labels:
                idx2label[(label['h'], label['t'])].append(label['r'])

            train_triple = list(idx2label.keys())
            j = 0
            for (h_idx, t_idx) in train_triple:
                if j == self.h_t_limit:
                    break

                hlist = item['vertexSet'][h_idx]
                tlist = item['vertexSet'][t_idx]
                if self.args.ablation == 1:
                    #co_sentence_filtering
                    co_sent_id = []
                    for hh in hlist:
                        for tt in tlist:
                            if hh['sent_id'] == tt['sent_id']:
                                co_sent_id.append(tt['sent_id'])
                    if len(co_sent_id) != 0:
                        new_hlist = []
                        new_tlist = []
                        for hh in hlist:
                            if hh['sent_id'] in co_sent_id:
                                new_hlist.append(hh)
                        for tt in tlist:
                            if tt['sent_id'] in co_sent_id:
                                new_tlist.append(tt)
                        hlist = new_hlist
                        tlist = new_tlist
                elif self.args.ablation == 2:
                    co_sent_id = []
                    for hh in hlist:
                        for tt in tlist:
                            if hh['sent_id'] == tt['sent_id']:
                                co_sent_id.append(tt['sent_id'])
                    new_hlist = []
                    new_tlist = []
                    for hh in hlist:
                        if hh['sent_id'] not in co_sent_id:
                            new_hlist.append(hh)
                    for tt in tlist:
                        if tt['sent_id'] not in co_sent_id:
                            new_tlist.append(tt)
                    hlist = new_hlist
                    tlist = new_tlist

                hlist = [ ( starts_pos[h['pos'][0]],  ends_pos[h['pos'][1]-1]  )  for h in hlist if h['pos'][1] < 511 and ends_pos[h['pos'][1]-1]<511 ]
                tlist = [ ( starts_pos[t['pos'][0]],  ends_pos[t['pos'][1]-1]  )  for t in tlist if t['pos'][1] < 511 and ends_pos[t['pos'][1]-1]<511 ]

                if len(hlist)==0 or len(tlist)==0:
                    continue

                for h in hlist:
                    if self.args.start_end_token == 1:
                        h_mapping[i, j, h[0] - 1] = 1.0 / len(hlist)
                        mlm_mask[i, h[0] - 1: h[1] + 1] = 1
                    else:
                        h_mapping[i, j, h[0]:h[1]] = 1.0 / len(hlist) / (h[1] - h[0])
                        mlm_mask[i, h[0]: h[1]] = 1

                for t in tlist:
                    if self.args.start_end_token == 1:
                        t_mapping[i, j, t[0] - 1] = 1.0 / len(tlist)
                        mlm_mask[i, t[0] - 1: t[1] + 1] = 1
                    else:
                        t_mapping[i, j, t[0]:t[1]] = 1.0 / len(tlist) / (t[1] - t[0])
                        mlm_mask[i, t[0]: t[1]] = 1

                label = idx2label[(h_idx, t_idx)]
                for r in label:
                    if r in self.rel2id:
                        r_idx = self.rel2id[r]
                        if r_idx not in rel_list:
                            rel_list[r_idx] = []
                        rel_list[r_idx].append([i, j])
                idx2pair[(i, j)] = [h_idx, t_idx]

                j += 1

            if not self.args.add_none:
                max_h_t_cnt = max(max_h_t_cnt, len(train_triple))
            else:
                lower_bound = min(len(item['na_triple']), self.neg_multiple)
                sel_ins = random.sample(item['na_triple'], lower_bound)

                for (h_idx, t_idx) in sel_ins:
                    if j == self.h_t_limit:
                        break
                    hlist = item['vertexSet'][h_idx]
                    tlist = item['vertexSet'][t_idx]

                    if self.args.ablation == 1:
                        #co_sentence_filtering
                        co_sent_id = []
                        for hh in hlist:
                            for tt in tlist:
                                if hh['sent_id'] == tt['sent_id']:
                                    co_sent_id.append(tt['sent_id'])
                        if len(co_sent_id) != 0:
                            new_hlist = []
                            new_tlist = []
                            for hh in hlist:
                                if hh['sent_id'] in co_sent_id:
                                    new_hlist.append(hh)
                            for tt in tlist:
                                if tt['sent_id'] in co_sent_id:
                                    new_tlist.append(tt)
                            hlist = new_hlist
                            tlist = new_tlist
                    elif self.args.ablation == 2:
                        co_sent_id = []
                        for hh in hlist:
                            for tt in tlist:
                                if hh['sent_id'] == tt['sent_id']:
                                    co_sent_id.append(tt['sent_id'])
                        new_hlist = []
                        new_tlist = []
                        for hh in hlist:
                            if hh['sent_id'] not in co_sent_id:
                                new_hlist.append(hh)
                        for tt in tlist:
                            if tt['sent_id'] not in co_sent_id:
                                new_tlist.append(tt)
                        hlist = new_hlist
                        tlist = new_tlist

                    hlist = [ ( starts_pos[h['pos'][0]],  ends_pos[h['pos'][1]-1]  )  for h in hlist if h['pos'][1] < 511 and ends_pos[h['pos'][1]-1]<511 ]
                    tlist = [ ( starts_pos[t['pos'][0]],  ends_pos[t['pos'][1]-1]  )  for t in tlist if t['pos'][1] < 511 and ends_pos[t['pos'][1]-1]<511 ]

                    hlist = [x for x in hlist if x[0] < x[1]]
                    tlist = [x for x in tlist if x[0] < x[1]]

                    if len(hlist)==0 or len(tlist)==0:
                        continue

                    for h in hlist:
                        if self.args.start_end_token == 1:
                            h_mapping[i, j, h[0] - 1] = 1.0 / len(hlist)
                            mlm_mask[i, h[0] - 1 : h[1] + 1] = 1
                        else:
                            h_mapping[i, j, h[0]:h[1]] = 1.0 / len(hlist) / (h[1] - h[0])
                            mlm_mask[i, h[0] : h[1]] = 1

                    for t in tlist:
                        if self.args.start_end_token == 1:
                            t_mapping[i, j, t[0] - 1] = 1.0 / len(tlist)
                            mlm_mask[i, t[0] - 1 : t[1] + 1] = 1
                        else:
                            t_mapping[i, j, t[0]:t[1]] = 1.0 / len(tlist) / (t[1] - t[0])
                            mlm_mask[i, t[0] : t[1]] = 1

                    rel_list_none.append([i, j])
                    idx2pair[(i, j)] = [h_idx, t_idx]
                    j += 1

                max_h_t_cnt = max(max_h_t_cnt, len(train_triple) + lower_bound)

        for k in rel_list:
            random.shuffle(rel_list[k])
        random.shuffle(rel_list_none)
        rel_sum_not_none = sum([len(rel_list[k]) for k in rel_list])
        total_rel_sum = rel_sum_not_none + len(rel_list_none)

        relation_label = torch.LongTensor(total_rel_sum).zero_()
        relation_label_idx = torch.LongTensor(total_rel_sum, 2).zero_()
        pos_num = 0
        jj = -1 - len(rel_list_none)

        for k in rel_list:
            for j in range(len(rel_list[k])):
                if len(rel_list[k]) % 2 == 1 and j == len(rel_list[k]) - 1:
                    break
                relation_label[pos_num] = k
                relation_label_idx[pos_num] = torch.LongTensor(rel_list[k][j])
                pos_num += 1
            if len(rel_list[k]) % 2 == 1:
                relation_label[jj] = k
                relation_label_idx[jj] = torch.LongTensor(rel_list[k][-1])
                jj -= 1

        for j in range(len(rel_list_none)):
            relation_label[rel_sum_not_none + j] = 0
            relation_label_idx[rel_sum_not_none + j] = torch.LongTensor(rel_list_none[j])
        # print(relation_label)
        rel_mask_pos = torch.LongTensor(total_rel_sum, total_rel_sum).zero_()
        rel_mask_neg = torch.LongTensor(total_rel_sum, total_rel_sum).zero_()
        for i in range(total_rel_sum):
            if i >= pos_num:
                break
            neg = []
            pos = []
            for j in range(total_rel_sum):
                idx_1 = relation_label_idx[i].numpy().tolist()
                idx_2 = relation_label_idx[j].numpy().tolist()
                pair_idx_1 = idx2pair[tuple(idx_1)]
                pair_idx_2 = idx2pair[tuple(idx_2)]
                if idx_1[0] == idx_2[0]:
                    if pair_idx_1[0] == pair_idx_2[0] or pair_idx_1[1] == pair_idx_2[1]:
                        continue
                if relation_label[i] != relation_label[j] and idx_1 != idx_2:
                    neg.append(j)
                if relation_label[i] == relation_label[j] and idx_1 != idx_2:
                    if i % 2 == 0 and j == i + 1:
                        pos.append(j)
                    elif i % 2 == 1 and j == i - 1:
                        pos.append(j)

            if len(neg) > self.args.neg_sample_num:
                neg = random.sample(neg, self.args.neg_sample_num)
            for j in neg:
                rel_mask_neg[i, j] = 1
            for j in pos:
                rel_mask_pos[i, j] = 1

        return {'context_idxs': context_idxs,
                'h_mapping': h_mapping[ :, :max_h_t_cnt, :].contiguous(),
                't_mapping': t_mapping[:, :max_h_t_cnt, :].contiguous(),
                'relation_label': relation_label.contiguous(),
                'relation_label_idx': relation_label_idx.contiguous(),
                'context_masks': context_masks,
                'rel_mask_pos': rel_mask_pos,
                'rel_mask_neg': rel_mask_neg,
                'pos_num': torch.tensor([pos_num]).cuda(),
                'mlm_mask': mlm_mask,
                }

    def get_wiki_batch(self, batch):
        batch_size = len(batch)
        if batch_size == 0:
            return {}
        max_length = self.max_length
        h_t_limit = self.h_t_limit
        context_idxs = torch.LongTensor(batch_size, max_length).zero_()

        h_mapping = torch.Tensor(batch_size, h_t_limit, max_length).zero_()
        query_mapping = torch.Tensor(batch_size, max_length).zero_()
        mlm_mask = torch.Tensor(batch_size, max_length).zero_()

        context_masks = torch.LongTensor(batch_size, max_length).zero_()
        start_positions = torch.LongTensor(batch_size).zero_()
        end_positions = torch.LongTensor(batch_size).zero_()

        rel_mask_pos = torch.LongTensor(len(batch), self.h_t_limit).zero_()
        rel_mask_neg = torch.LongTensor(len(batch), self.h_t_limit).zero_()

        j = 0

        relation_label_idx = []
        relation_label = []
        for i in range(len(batch)):
            item = batch[i]
            context_idxs[i].copy_(torch.from_numpy(item[0]))
            context_masks[i].copy_(torch.from_numpy(item[1]))

            item = batch[i][3]

            starts_pos = batch[i][2][:, 0]
            ends_pos = batch[i][2][:, 1]

            hlist = item['vertexSet'][item['label_hop']['h']]
            hlist = [ ( starts_pos[h['pos'][0]],  ends_pos[h['pos'][1]-1]  )  for h in hlist if h['pos'][1] < 511 and ends_pos[h['pos'][1]-1]<511 ]
            hlist = [x for x in hlist if x[0] < x[1]]
            if len(hlist)==0:
                continue
            for h in hlist:
                if self.args.start_end_token == 1:
                    query_mapping[i, h[0] - 1] = 1.0 / len(hlist)
                    mlm_mask[i, h[0] - 1 : h[1] + 1] = 1
                else:
                    query_mapping[i, h[0]:h[1]] = 1.0 / len(hlist) / (h[1] - h[0])
                    mlm_mask[i, h[0] : h[1]] = 1

            flag = 0
            for vertex_idx, vertex in enumerate(item['vertexSet']):
                if vertex_idx == item['label_hop']['h']:
                    continue
                hlist = item['vertexSet'][vertex_idx]
                hlist = [ ( starts_pos[h['pos'][0]],  ends_pos[h['pos'][1]-1]  )  for h in hlist if h['pos'][1] < 511 and ends_pos[h['pos'][1]-1]<511]
                hlist = [x for x in hlist if x[0] < x[1]]
                if len(hlist)==0:
                    continue

                if vertex_idx == item['label_hop']['t']:
                    rel_mask_pos[i, j] = 1.0
                    rel_mask_neg[i, j] = 1.0
                    answer = random.choice(hlist)
                    start_positions[i] = answer[0].tolist()
                    end_positions[i] = answer[1].tolist() - 1
                    flag = 1
                elif j == self.h_t_limit:
                    continue
                else:
                    rel_mask_neg[i, j] = 1.0
                    flag = 0

                for h in hlist:
                    if self.args.start_end_token == 1:
                        h_mapping[i, j, h[0] - 1] = 1.0 / len(hlist)
                        mlm_mask[i, h[0] - 1 : h[1] + 1] = 1
                    else:
                        h_mapping[i, j, h[0]:h[1]] = 1.0 / len(hlist) / (h[1] - h[0])
                        mlm_mask[i, h[0]: h[1]] = 1

                relation_label_idx.append([i, j])
                j += 1

        relation_label_idx = torch.LongTensor(relation_label_idx)
        return {'context_idxs': context_idxs,
                'h_mapping': h_mapping[: , : j, :].contiguous(),
                'query_mapping': query_mapping.contiguous(),
                'context_masks': context_masks,
                'rel_mask_pos': rel_mask_pos[:, :j],
                'rel_mask_neg': rel_mask_neg[:, :j],
                'relation_label_idx': relation_label_idx,
                'start_positions': start_positions,
                'end_positions': end_positions,
                'mlm_mask': mlm_mask,
                }

    def get_train_batch(self, batch):
        batch_doc = self.get_doc_batch([b[0] for b in batch])
        batch_wiki = self.get_wiki_batch([b[1] for b in batch])
        return [batch_doc, batch_wiki]
