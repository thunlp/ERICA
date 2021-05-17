import argparse
import glob
import logging
import os
import random
import timeit

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, Dataset
import string
from transformers import (
    BertModel,
    BertPreTrainedModel,
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    BertConfig,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForQuestionAnswering,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaModel,
    RobertaForQuestionAnswering,
    RobertaTokenizer,
    XLMConfig,
    XLMForQuestionAnswering,
    XLMTokenizer,
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
import json
# from triviaqa_utils import evaluation_utils

import shutil

class RobertaForWikihopMulti(BertPreTrainedModel):
    base_model_prefix = "roberta"
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        answer_index=None,
        instance_mask=None,
        candidate_pos=None,
        candidate_num=None,
    ):
        torch.set_printoptions(profile="full")
        # logger.info(input_ids.size())
        # logger.info(input_ids)
        # logger.info(attention_mask.size())
        # logger.info(attention_mask)
        # logger.info(answer_index.size())
        # logger.info(answer_index)
        # logger.info(instance_mask.size())
        # logger.info(instance_mask)
        # logger.info(candidate_pos.size())
        # logger.info(candidate_pos)
        # logger.info(candidate_num.size())
        # logger.info(candidate_num)
        # logger.info('\n')

        bsz = input_ids.shape[0]
        max_segment = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        candidate_pos = candidate_pos.view(-1, candidate_pos.size()[2], candidate_pos.size()[3])

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs[0]

        context_hidden = torch.matmul(candidate_pos, sequence_output)

        logits = self.qa_outputs(context_hidden).squeeze(-1)
        logits = logits.view(bsz, max_segment, -1)

        ignore_index = -1
        candidate_mask = torch.sum(candidate_pos, dim = 2)
        candidate_mask = candidate_mask.view(bsz, max_segment, -1)
        ignore_mask = candidate_mask > 0
        ignore_mask = 1 - ignore_mask.long()
        ignore_mask = ignore_mask.bool()
        logits[ignore_mask] = 0
        logits = torch.sum(logits, dim = 1)

        candidate_mask = torch.sum(candidate_mask, dim = 1)
        ignore_mask = candidate_mask > 0
        ignore_mask = 1 - ignore_mask.long()
        ignore_mask = ignore_mask.bool()

        logits[ignore_mask] = -10000.0

        outputs = (logits,) + outputs[2:]
        if answer_index is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, answer_index)
            outputs = (loss,) + outputs

        return outputs

class BertForWikihopMulti(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForWikihopMulti, self).__init__(config)
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        answer_index=None,
        instance_mask=None,
        candidate_pos=None,
        candidate_num=None,
    ):
        torch.set_printoptions(profile="full")
        # logger.info(input_ids.size())
        # logger.info(attention_mask.size())
        # logger.info(token_type_ids.size())
        # logger.info(answer_index.size())
        # logger.info(instance_mask.size())
        # logger.info(candidate_pos.size())
        # logger.info(candidate_num.size())
        # logger.info('\n')

        bsz = input_ids.shape[0]
        max_segment = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        candidate_pos = candidate_pos.view(-1, candidate_pos.size()[2], candidate_pos.size()[3])

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        context_hidden = torch.matmul(candidate_pos, sequence_output)

        logits = self.qa_outputs(context_hidden).squeeze(-1)
        logits = logits.view(bsz, max_segment, -1)

        ignore_index = -1
        candidate_mask = torch.sum(candidate_pos, dim = 2)
        candidate_mask = candidate_mask.view(bsz, max_segment, -1)
        ignore_mask = candidate_mask > 0
        ignore_mask = 1 - ignore_mask.long()
        ignore_mask = ignore_mask.bool()
        logits[ignore_mask] = 0
        logits = torch.sum(logits, dim = 1)

        candidate_mask = torch.sum(candidate_mask, dim = 1)
        ignore_mask = candidate_mask > 0
        ignore_mask = 1 - ignore_mask.long()
        ignore_mask = ignore_mask.bool()

        logits[ignore_mask] = -10000.0

        outputs = (logits,) + outputs[2:]
        if answer_index is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, answer_index)
            outputs = (loss,) + outputs

        return outputs

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForWikihopMulti, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForWikihopMulti, RobertaTokenizer)
}

TEST_SPEED = False

class WikihopDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_len, max_doc_len, doc_stride,
                 max_num_candidates, ignore_seq_with_no_answers, max_question_len, max_segment, train_rl=False, race_npy_path=None, model_type=None):
        assert os.path.isfile(file_path)
        self.file_path = file_path
        with open(self.file_path, "r", encoding='utf-8') as f:
            print(f'reading file: {self.file_path}')
            self.data_json = json.load(f)
            # self.data_json = self.data_json[:50]
            print(f'done reading file: {self.file_path}')
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_doc_len = max_doc_len
        self.doc_stride = doc_stride
        self.max_num_candidates = max_num_candidates
        self.ignore_seq_with_no_answers = ignore_seq_with_no_answers
        self.max_question_len = max_question_len
        self.max_segment = max_segment
        self.zero_item = [0] * self.max_seq_len
        self.one_item = [1] * self.max_seq_len
        self.one_zero_item = [1] + [0] * (self.max_seq_len-1)
        self.model_type = model_type

        # self.non_candidate = [-1] * self.max_seq_len
        self.train_rl = train_rl
        if train_rl:
            file_name = "npy_folder/wikihop_"+str( self.max_seq_len )+"_"+str( self.max_segment )+".memmap"
            self.guide_file = np.memmap(filename = file_name, shape=(len(self.data_json),  self.max_segment, 8, self.max_seq_len), mode='r', dtype=np.float32)

    def __len__(self):
        return len(self.data_json)

    def __getitem__(self, idx):
        entry = self.data_json[idx]
        if self.model_type == 'bert':
            return self.one_example_to_tensors(entry, idx)
        elif self.model_type == 'roberta':
            return self.one_example_to_tensors_roberta(entry, idx)

    def one_example_to_tensors_roberta(self, example, idx):
        torch.set_printoptions(profile="full")
        # def tokenize(text):
        #     D = {'___MASK' + str(x) + '___': '[unused' + str(x) + ']' for x in range(1, 101)}
        #     text_tokens = []
        #     textraw = [text]
        #     for delimiter in D:
        #         ntextraw = []
        #         for i in range(len(textraw)):
        #             t = textraw[i].split(delimiter)
        #             for j in range(len(t)):
        #                 ntextraw += [t[j]]
        #                 if j != len(t)-1:
        #                     ntextraw += [D[delimiter]]
        #         textraw = ntextraw
        #     return textraw
        def tokenize(text):
            D = {'___MASK' + str(x) + '___': ['<s> ' + str(x) + ' </s>'] for x in range(1, 101)}
            text_tokens = []
            textraw = [text]
            for delimiter in D:
                ntextraw = []
                for i in range(len(textraw)):
                    t = textraw[i].split(delimiter)
                    for j in range(len(t)):
                        ntextraw += [t[j]]
                        if j != len(t)-1:
                            ntextraw += D[delimiter]
                textraw = ntextraw
            return textraw

        qid = example['id']
        question_text = example['query']

        def ff(x):
            if len(x) == 0:
                return ''
            if len(x) == 1:
                return x[0].upper()
            return x[0].upper() + x[1: ]
        question_text = question_text.lower()
        question_text_1 = ' '.join(question_text.split(' ')[0].split('_'))
        question_text_2 = ' '.join(ff(x) for x in question_text.split(' ')[1: ])
        question_text = question_text_1 + ' ' + question_text_2 + '</s>'
        # logger.info(question_text)
        candidates = example['candidates']
        supports = example['supports']
        new_candidates = []
        for c in candidates:
            for s in supports:
                if c in s:
                    new_candidates.append(c)
                    break
        candidates = new_candidates
        answer = example['answer']

        context_start = '[unused102] '

        #query_tokens = [self.tokenizer.cls_token] + tokenize(question_text, self.tokenizer)
        query_tokens = [self.tokenizer.cls_token] + self.tokenizer.tokenize(''.join(tokenize(question_text)))

        heads = query_tokens
        # logger.info(heads)
        # logger.info([self.tokenizer.convert_tokens_to_ids(x) for x in heads])

        answer_index = -1
        for cand_idx in range(len(candidates)):
            if candidates[cand_idx]==answer:
                answer_index = cand_idx
        assert( answer_index >= 0 )

        all_doc_tokens = []
        for support in supports:
            all_doc_tokens += [context_start] + tokenize(support)
        all_doc_tokens = self.tokenizer.tokenize(''.join(all_doc_tokens[: self.max_doc_len]))

        max_tokens_per_doc_slice = self.max_seq_len - len(heads) - 1
        assert max_tokens_per_doc_slice > 0
        if self.doc_stride < 0:
            # negative doc_stride indicates no sliding window, but using first slice
            self.doc_stride = -100 * len(all_doc_tokens)  # large -ve value for the next loop to execute once

        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []
        instance_mask = []
        candidate_pos = []
        candidate_num = [len(candidates)]
        cnt = 0
        candidate_tokens = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('<s> ' + str(c[7: -3]) + ' </s>')) for c in candidates]
        #candidate_tokens = [self.tokenizer.convert_tokens_to_ids(tokenize(c, self.tokenizer)) for c in candidates]
        for slice_start in range(0, len(all_doc_tokens), max_tokens_per_doc_slice - self.doc_stride):
            slice_end = min(slice_start + max_tokens_per_doc_slice, len(all_doc_tokens))
            doc_slice_tokens = all_doc_tokens[slice_start:slice_end]
            tokens =  heads + doc_slice_tokens + [self.tokenizer.sep_token]
            segment_ids = [0] * (len(heads)) + [1] * (len(doc_slice_tokens) + 1)
            assert len(segment_ids) == len(tokens)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding_len = self.max_seq_len - len(input_ids)
            input_ids.extend([self.tokenizer.pad_token_id] * padding_len)

            input_mask.extend([0] * padding_len)
            segment_ids.extend([0] * padding_len)
            candidate_ids = [[0.] * self.max_seq_len for _ in range(self.max_num_candidates)]
            for c_idx in range(len(candidate_tokens)):
                c_token = candidate_tokens[c_idx]
                x = 0
                pos_list = []
                while(x <= len(input_ids) - len(c_token)):
                    flag = 0
                    for y in range(len(c_token)):
                        if input_ids[x + y] != c_token[y]:
                            flag = 1
                    if flag == 0:
                        for _ in range(len(c_token)):
                            pos_list.append(x)
                            #candidate_ids[c_idx].append(1.0 / len(c_token))
                            x += 1
                    else:
                        #candidate_ids[c_idx].append(0.)
                        x += 1
                for xx in pos_list:
                    # input_ids[xx] = 0
                    candidate_ids[c_idx][xx] = 1.0
                # if np.sum(candidate_ids[c_idx]) > 0:
                #     logger.info(np.sum(candidate_ids[c_idx]))
                #     logger.info(candidate_ids[c_idx])
                #     logger.info('\n')
            # logger.info(question_text)
            # logger.info([self.tokenizer.decode(x) for x in input_ids])
            # logger.info(input_ids)
            # logger.info('\n')

            assert len(input_ids) == self.max_seq_len
            assert len(input_mask) == self.max_seq_len
            assert len(segment_ids) == self.max_seq_len

            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)
            instance_mask.append(1)
            candidate_pos.append(candidate_ids)

            cnt += 1
            if cnt >= self.max_segment:
                break

        while cnt < self.max_segment:
            input_ids_list.append(self.one_item)
            input_mask_list.append(self.one_zero_item)  # avoid NAN
            segment_ids_list.append(self.zero_item)
            instance_mask.append(0)
            candidate_pos.append([[0.] * self.max_seq_len for _ in range(self.max_num_candidates)])
            cnt += 1
        candidate_pos = torch.tensor(candidate_pos, dtype=torch.float32)
        candidate_pos_num = torch.sum(candidate_pos, dim = 2, keepdim = True)
        candidate_nonzero = candidate_pos_num > 0
        candidate_nonzero = torch.sum(candidate_nonzero, dim = 0, keepdim = True)
        #candidate_pos_num = torch.sum(candidate_pos, dim = 0, keepdim = True)
        #candidate_pos_num = torch.sum(candidate_pos_num, dim = 2, keepdim = True)
        for idx_1 in range(candidate_pos_num.size()[0]):
            for idx_2 in range(candidate_pos_num.size()[1]):
                if candidate_pos_num[idx_1, idx_2, 0] == 0:
                    candidate_pos_num[idx_1, idx_2, 0] = 1
        for idx in range(candidate_pos_num.size()[1]):
            if candidate_nonzero[0, idx, 0] == 0:
                candidate_nonzero[0, idx, 0] = 1
        candidate_pos = candidate_pos / candidate_pos_num / candidate_nonzero

        item = [torch.tensor(input_ids_list),
                                torch.tensor(input_mask_list),
                                torch.tensor(segment_ids_list),
                                torch.tensor(instance_mask),
                                torch.tensor(answer_index),#.view(-1)
                                candidate_pos,
                                torch.tensor(candidate_num),
                                ]
        if self.train_rl:
            item.append( torch.from_numpy(self.guide_file[idx]) )
        return item

    def one_example_to_tensors(self, example, idx):
        torch.set_printoptions(profile="full")
        def tokenize(text, tokenizer):
            D = {'___MASK' + str(x) + '___': '[unused' + str(x) + ']' for x in range(1, 101)}
            text_tokens = []
            textraw = [text]
            for delimiter in D:
                ntextraw = []
                for i in range(len(textraw)):
                    t = textraw[i].split(delimiter)
                    for j in range(len(t)):
                        ntextraw += [t[j]]
                        if j != len(t)-1:
                            ntextraw += [D[delimiter]]
                textraw = ntextraw

            text = []
            for t in textraw:
                if t in list(D.values()):
                    text += [t]
                else:
                    tokens = tokenizer.tokenize(t)
                    for tok in tokens:
                        text += [tok]
            return text

        qid = example['id']
        question_text = example['query']
        candidates = example['candidates']
        supports = example['supports']
        new_candidates = []
        for c in candidates:
            for s in supports:
                if c in s:
                    new_candidates.append(c)
                    break
        candidates = new_candidates
        answer = example['answer']

        context_start = '[unused102]'

        query_tokens = [self.tokenizer.cls_token] + tokenize(question_text, self.tokenizer)

        heads = query_tokens

        answer_index = -1
        for cand_idx in range(len(candidates)):
            if candidates[cand_idx]==answer:
                answer_index = cand_idx
        assert( answer_index >= 0 )

        all_doc_tokens = []
        for support in supports:
            all_doc_tokens += [context_start] + tokenize(support, self.tokenizer)
        all_doc_tokens = all_doc_tokens[: self.max_doc_len]

        max_tokens_per_doc_slice = self.max_seq_len - len(heads) - 1
        assert max_tokens_per_doc_slice > 0
        if self.doc_stride < 0:
            # negative doc_stride indicates no sliding window, but using first slice
            self.doc_stride = -100 * len(all_doc_tokens)  # large -ve value for the next loop to execute once

        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []
        instance_mask = []
        candidate_pos = []
        candidate_num = [len(candidates)]
        cnt = 0
        candidate_tokens = [self.tokenizer.convert_tokens_to_ids(tokenize(c, self.tokenizer)) for c in candidates]
        for slice_start in range(0, len(all_doc_tokens), max_tokens_per_doc_slice - self.doc_stride):
            slice_end = min(slice_start + max_tokens_per_doc_slice, len(all_doc_tokens))
            doc_slice_tokens = all_doc_tokens[slice_start:slice_end]
            tokens =  heads + doc_slice_tokens + [self.tokenizer.sep_token]
            segment_ids = [0] * (len(heads)) + [1] * (len(doc_slice_tokens) + 1)
            assert len(segment_ids) == len(tokens)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding_len = self.max_seq_len - len(input_ids)
            input_ids.extend([self.tokenizer.pad_token_id] * padding_len)
            input_mask.extend([0] * padding_len)
            segment_ids.extend([0] * padding_len)

            candidate_ids = [[0.] * self.max_seq_len for _ in range(self.max_num_candidates)]
            for c_idx in range(len(candidate_tokens)):
                c_token = candidate_tokens[c_idx]
                x = 0
                pos_list = []
                while(x <= len(input_ids) - len(c_token)):
                    flag = 0
                    for y in range(len(c_token)):
                        if input_ids[x + y] != c_token[y]:
                            flag = 1
                    if flag == 0:
                        for _ in range(len(c_token)):
                            pos_list.append(x)
                            #candidate_ids[c_idx].append(1.0 / len(c_token))
                            x += 1
                    else:
                        #candidate_ids[c_idx].append(0.)
                        x += 1
                for xx in pos_list:
                    candidate_ids[c_idx][xx] = 1.0

                # if np.sum(candidate_ids[c_idx]) > 0:
                #     logger.info(np.sum(candidate_ids[c_idx]))
                #     logger.info(candidate_ids[c_idx])
                #     logger.info('\n')

            assert len(input_ids) == self.max_seq_len
            assert len(input_mask) == self.max_seq_len
            assert len(segment_ids) == self.max_seq_len

            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)
            instance_mask.append(1)
            candidate_pos.append(candidate_ids)

            cnt += 1
            if cnt >= self.max_segment:
                break

        while cnt < self.max_segment:
            input_ids_list.append(self.zero_item)
            input_mask_list.append(self.one_zero_item)  # avoid NAN
            segment_ids_list.append(self.zero_item)
            instance_mask.append(0)
            candidate_pos.append([[0.] * self.max_seq_len for _ in range(self.max_num_candidates)])
            cnt += 1
        candidate_pos = torch.tensor(candidate_pos, dtype=torch.float32)
        candidate_pos_num = torch.sum(candidate_pos, dim = 2, keepdim = True)
        candidate_nonzero = candidate_pos_num > 0
        candidate_nonzero = torch.sum(candidate_nonzero, dim = 0, keepdim = True)
        #candidate_pos_num = torch.sum(candidate_pos, dim = 0, keepdim = True)
        #candidate_pos_num = torch.sum(candidate_pos_num, dim = 2, keepdim = True)
        for idx_1 in range(candidate_pos_num.size()[0]):
            for idx_2 in range(candidate_pos_num.size()[1]):
                if candidate_pos_num[idx_1, idx_2, 0] == 0:
                    candidate_pos_num[idx_1, idx_2, 0] = 1
        for idx in range(candidate_pos_num.size()[1]):
            if candidate_nonzero[0, idx, 0] == 0:
                candidate_nonzero[0, idx, 0] = 1
        candidate_pos = candidate_pos / candidate_pos_num / candidate_nonzero

        item = [torch.tensor(input_ids_list),
                                torch.tensor(input_mask_list),
                                torch.tensor(segment_ids_list),
                                torch.tensor(instance_mask),
                                torch.tensor(answer_index),#.view(-1)
                                candidate_pos,
                                torch.tensor(candidate_num),
                                ]
        if self.train_rl:
            item.append( torch.from_numpy(self.guide_file[idx]) )
        return item

    @staticmethod
    def collate_one_doc_and_lists(batch):
        # num_metadata_fields = 0
        fields = [x for x in zip(*batch)]
        # stacked_fields = [torch.stack(field) for field in fields[:-num_metadata_fields]]  # don't stack metadata fields
        # stacked_fields.extend(fields[-num_metadata_fields:])  # add them as lists not torch tensors

        stacked_fields = [torch.stack(field) for field in fields]
        max_candidate_num = torch.max(stacked_fields[6])
        stacked_fields[5] = stacked_fields[5][:, :, :max_candidate_num, :]

        global TEST_SPEED
        if TEST_SPEED:
            vaild = torch.sum(stacked_fields[3], dim=0)
            for j in range(stacked_fields[0].shape[1])[::-1]:
                if vaild[j]!=0:
                    break
            stacked_fields[0] = stacked_fields[0][:,0: j+1,:]
            stacked_fields[1] = stacked_fields[1][:,0: j+1,:]
            stacked_fields[2] = stacked_fields[2][:,0: j+1,:]
            stacked_fields[3] = stacked_fields[3][:,0: j+1]
            stacked_fields[5] = stacked_fields[5][:,0: j+1, :, :]

        # for x in stacked_fields:
        #     logger.info(x.size())
        # logger.info('\n')

        return stacked_fields


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()



def train(args, model, tokenizer):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size #* max(1, args.n_gpu)

    train_dataset = WikihopDataset(file_path=args.train_dataset, tokenizer=tokenizer,
                                max_seq_len=args.max_seq_len, max_doc_len=args.max_doc_len,
                                doc_stride=args.doc_stride,
                                max_num_candidates=args.max_num_candidates,
                                max_question_len=args.max_question_len,
                                ignore_seq_with_no_answers=args.ignore_seq_with_no_answers,
                                max_segment=args.max_segment, model_type=args.model_type)

    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.local_rank != -1  else RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=(sampler is None),
                                    sampler=sampler,  num_workers=1,
                                    collate_fn=WikihopDataset.collate_one_doc_and_lists)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1*t_total), num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)
    best_acc = 0
    epoch_idx = 0

    for _ in train_iterator:
        epoch_idx += 1
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            #for step, batch in enumerate(train_dataloader):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": None if args.model_type in ["xlm", "roberta", "distilbert"] else batch[2],
                "instance_mask": batch[3],
                "answer_index": batch[4],
                "candidate_pos": batch[5],
                "candidate_num": batch[6],
            }

            # if args.model_type in ["xlnet", "xlm"]:
            #     inputs.update({"cls_index": batch[5], "p_mask": batch[6]})

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training


            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    logging_loss = tr_loss

                # Save model checkpoint
                if args.local_rank in [-1, 0]  and (args.save_steps > 0 and (global_step % args.save_steps == 0 or global_step==10)):
                    if args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        acc = results['acc']
                        print ('acc:', acc)

                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    # if not os.path.exists(output_dir):
                    #     os.makedirs(output_dir)
                    # # Take care of distributed/parallel training
                    # model_to_save = model.module if hasattr(model, "module") else model
                    # model_to_save.save_pretrained(output_dir)
                    # tokenizer.save_pretrained(output_dir)

                    # torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    # logger.info("Saving model checkpoint to %s", output_dir)

                    # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    # logger.info("Saving optimizer and scheduler states to %s", output_dir)
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        results = evaluate(args, model, tokenizer)
        acc = results['acc']
        print('epoch:' + str(epoch_idx))
        print('acc:', acc)
        if acc > best_acc:
            best_acc = acc
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    print('best_acc: ' + str(best_acc))
    if global_step == 0:
        print('no_train!')
        return 0, 0
    else:
        return global_step, tr_loss / global_step

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def evaluate(args, model, tokenizer, prefix=""):
    # dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
    global TEST_SPEED
    TEST_SPEED = True
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    dataset = WikihopDataset(file_path=args.dev_dataset, tokenizer=tokenizer,
                                max_seq_len=args.max_seq_len, max_doc_len=args.max_doc_len,
                                doc_stride=args.doc_stride,
                                max_num_candidates=args.max_num_candidates,
                                max_question_len=args.max_question_len,
                                ignore_seq_with_no_answers=args.ignore_seq_with_no_answers,
                                max_segment=args.max_segment, model_type=args.model_type)

    # Note that DistributedSampler samples randomly
    eval_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if args.local_rank != -1  else SequentialSampler(dataset)

    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=1,
                                    collate_fn=WikihopDataset.collate_one_doc_and_lists)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)


    eval_accuracy = 0
    eval_examples = 0
    # tot_loss = 0
    #macs_list = json.load(open('macs_list.json'))
    flops = 0
    bert_flops = 0
    bert_512_flops = 0

    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
        #for batch in eval_dataloader:

        model.eval()

        batch = [t.to(args.device) for t in batch]
        max_segment = batch[1].shape[1]
        L_s = batch[1].view(-1, batch[1].shape[-1]).sum(dim=1)
        l = int(torch.max(L_s))
        batch[0] = batch[0][:, :, :l]
        batch[1] = batch[1][:, :, :l]
        batch[2] = batch[2][:, :, :l]

        with torch.no_grad():

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": None if args.model_type in ["xlm", "roberta", "distilbert"] else batch[2],
                "instance_mask": batch[3],
                # "answer_index": batch[4],
                "candidate_pos": batch[5],
                "candidate_num": batch[6],
            }

            outputs = model(**inputs)
            logits = outputs[0]

        # tot_loss += loss.item()

        logits = to_list(logits)
        label_ids = to_list(batch[4])

        tmp_eval_accuracy = accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        eval_examples += len(logits)

        hidden_size = 768#model.hidden_size if hasattr(model, "hidden_size") else model.module.hidden_size
        num_labels = 1#model.num_labels if hasattr(model, "num_labels") else model.module.num_labels

    #print ('Flops:', flops / len(dataset) / 1000000.0)
    #print ('BERT FLOPS:', 2*bert_flops/len(dataset)/1000000.0)

    acc = eval_accuracy / eval_examples
    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    TEST_SPEED = False

    return {'acc': acc}#, 'loss': tot_loss/eval_examples}#, 'sum_acc': eval_accuracy, 'ge': eval_examples, 'loss': tot_loss / eval_examples}#, 'subword_em': sum(all_subword_em) / ge }




def evaluate_grad(args, model, tokenizer, prefix=""):
    # dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    args.dev_dataset = os.path.join(args.data_dir, args.predict_file)

    dataset = WikihopDataset(file_path=args.train_dataset, tokenizer=tokenizer,
                                max_seq_len=args.max_seq_len, max_doc_len=args.max_doc_len,
                                doc_stride=args.doc_stride,
                                max_num_candidates=args.max_num_candidates,
                                max_question_len=args.max_question_len,
                                ignore_seq_with_no_answers=args.ignore_seq_with_no_answers,
                                max_segment=args.max_segment, model_type=args.model_type)

    # Note that DistributedSampler samples randomly
    eval_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if args.local_rank != -1  else SequentialSampler(dataset)

    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=1,
                                    collate_fn=WikihopDataset.collate_one_doc_and_lists)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)


    eval_accuracy = 0
    eval_examples = 0
    tot_loss = 0
    output_size = 8
    max_segment = args.max_segment
    max_seq_length = args.max_seq_len


    file_name = "npy_folder/wikihop_"+str( max_seq_length)+"_"+str(max_segment)+".memmap"
    layers_scores = np.memmap(filename = file_name, shape=(len(dataset),  max_segment, 8, max_seq_length), mode='w+', dtype=np.float32)

    cnt = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
        #for batch in eval_dataloader:
        model.eval()

        batch = tuple(t.to(args.device) for t in batch)

        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": None if args.model_type in ["xlm", "roberta", "distilbert"] else batch[2],
            "instance_mask": batch[3],
            "answer_index": batch[4],
            "candidate_pos": batch[5],
            "candidate_num": batch[6],
        }

        outputs = model(**inputs)
        loss, logits = outputs[:2]

        hidden_states = outputs[2]
        hidden_states = hidden_states[1:]

        last_val = hidden_states[8]
        # last_val_2 = hidden_states[9]
        # last_val_3 = hidden_states[10]

        grads = torch.autograd.grad(loss, [last_val,])
        grad_delta = grads[0]
        # grad_delta_2 = grads[1]
        # grad_delta_3 = grads[2]
        for idx in range(output_size):
            with torch.no_grad():
                delta = last_val - hidden_states[idx]
                dot = torch.einsum("bli,bli->bl", [grad_delta, delta])
                score = dot.abs()  # (bsz, seq_len)
                score = score.view(-1, max_segment, max_seq_length).detach()

            score = score.cpu().numpy()
            layers_scores[cnt: cnt+score.shape[0], :,idx, :] = score

        cnt += score.shape[0]

        tot_loss += loss.item()

        logits = to_list(logits)
        label_ids = to_list(batch[4])

        tmp_eval_accuracy = accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        eval_examples += len(logits)


    acc = eval_accuracy / eval_examples
    print (acc)
    return {'acc': acc}


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: ",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default="data",
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default="train.json",
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default="dev.json",
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="cachedir",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=5.0, type=float, help="Total number of training epochs to perform."
    )

    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )

    parser.add_argument("--logging_steps", type=int, default=5, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=-1, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")

    parser.add_argument("--do_eval_grad",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--race_npy_path",
                        default="npy_folder/wikihop_test.npy",
                        type=str,
                        required=False)

    parser.add_argument("--train_rl",
                        default=False,
                        action='store_true',
                        help=" ")

    parser.add_argument("--train_both",
                        default=False,
                        action='store_true',
                        help=" ")

    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Maximum length of seq passed to the transformer model")
    parser.add_argument("--max_doc_len", type=int, default=512,
                        help="Maximum number of wordpieces of the input document")
    parser.add_argument("--max_num_candidates", type=int, default=79,
                        help="")
    parser.add_argument("--max_question_len", type=int, default=55,
                        help="Maximum length of the question")
    parser.add_argument("--doc_stride", type=int, default=-1,
                        help="Overlap between document chunks. Use -1 to only use the first chunk")
    parser.add_argument("--ignore_seq_with_no_answers", action='store_true',
                        help="each example should have at least one answer. Default is False")
    parser.add_argument("--test", action='store_true', help="Test only, no training")
    parser.add_argument("--max_segment", type=int, default=8, help="8 for 512 bert")

    parser.add_argument("--alpha", default=1.0, type=float, help="")
    parser.add_argument("--guide_rate", default=0.5, type=float, help="")
    parser.add_argument("--ckpt_to_load", default='None', type=str, help="")

    args = parser.parse_args()

    args.train_dataset = os.path.join(args.data_dir, args.train_file)
    args.dev_dataset = os.path.join(args.data_dir, args.predict_file)


    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.do_eval_grad:
        config.output_hidden_states = True

    if args.model_type == 'bert':
        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
    elif args.model_type == 'roberta':
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
        special_tokens_dict = {'additional_special_tokens': ['[unused' + str(x) + ']' for x in range(1, 105)]}
        tokenizer.add_special_tokens(special_tokens_dict)
        print(tokenizer.additional_special_tokens)
        print(tokenizer.additional_special_tokens_ids)


    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if args.model_type == 'bert':
        if args.ckpt_to_load != "None":
            print("********* load from ckpt/"+args.ckpt_to_load+" ***********")
            if os.path.exists('***path_to_your_bert_model***'):
                ckpt = torch.load("***path_to_your_trained_model***"+args.ckpt_to_load)
            else:
                ckpt = torch.load("***path_to_your_trained_model***"+args.ckpt_to_load)
            model.bert.load_state_dict(ckpt["bert-base"])
        else:
            print("*******No ckpt to load, Let's use bert base!*******")
    elif args.model_type == 'roberta':
        if args.ckpt_to_load != "None":
            print("********* load from ckpt/"+args.ckpt_to_load+" ***********")
            if os.path.exists('***path_to_your_roberta_model***'):
                ckpt = torch.load("***path_to_your_trained_model***"+args.ckpt_to_load)
            else:
                ckpt = torch.load("***path_to_your_trained_model***"+args.ckpt_to_load)
            model.roberta.load_state_dict(ckpt["bert-base"])
        else:
            print("*******No ckpt to load, Let's use roberta base!*******")
        model.roberta.resize_token_embeddings(len(tokenizer))

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    if args.do_train:
        if args.train_rl:
            # train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False, train_pred=True)
            global_step, tr_loss = train_rl(args , model, tokenizer)
        elif args.train_both:
            # train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
            global_step, tr_loss = train_both(args, model, tokenizer)
        else:
            # train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
            global_step, tr_loss = train(args, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        # logger.info("Saving model checkpoint to %s", args.output_dir)
        # # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # # They can then be reloaded using `from_pretrained()`
        # # Take care of distributed/parallel training
        # model_to_save = model.module if hasattr(model, "module") else model
        # model_to_save.save_pretrained(args.output_dir)
        # tokenizer.save_pretrained(args.output_dir)

        # # Good practice: save your training arguments together with the trained model
        # #torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # # Load a trained model and vocabulary that you have fine-tuned
        # model = model_class.from_pretrained(args.output_dir, force_download=True)
        # tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        # model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        # if args.do_train:
        logger.info("Loading checkpoints saved during training for evaluation")
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
        # else:
        #     logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
        #     checkpoints = [args.model_name_or_path]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, force_download=True)
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model, tokenizer, prefix=global_step)
            print (result)
            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(results))

    if args.do_eval_grad and args.local_rank in [-1, 0]:
        logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
        checkpoints = [args.output_dir]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, force_download=True, config=config)
            model.to(args.device)

            # Evaluate
            result = evaluate_grad(args, model, tokenizer, prefix=global_step)




    return results


if __name__ == "__main__":
    main()
