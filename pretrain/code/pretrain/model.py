import os
import pdb
import torch
import torch.nn as nn
from pytorch_metric_learning.losses.ntxent_loss import NTXentLoss, GenericPairLoss
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from transformers import (BertForMaskedLM, BertTokenizer,
                            RobertaForMaskedLM, RobertaTokenizer,)
import random
from torch.nn import CrossEntropyLoss
import numpy as np

def mask_tokens(inputs, tokenizer, not_mask_pos=None):
    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, 0.15)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    if not_mask_pos is None:
        masked_indices = torch.bernoulli(probability_matrix).bool()
    else:
        masked_indices = torch.bernoulli(probability_matrix).bool() & (~(not_mask_pos.bool())) # ** can't mask entity marker **
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs.cuda(), labels.cuda()

class NTXentLoss_doc(GenericPairLoss):

    def __init__(self, temperature, **kwargs):
        super().__init__(**kwargs, use_similarity=True, mat_based_loss=False)
        self.temperature = temperature

    def forward(self, embeddings, labels, indices_tuple=None, pos_num=None):
        labels = labels.to(embeddings.device)
        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        loss = self.compute_loss(embeddings, labels, indices_tuple, pos_num)
        return loss

    def compute_loss(self, embeddings, labels, indices_tuple, pos_num):
        mat = lmu.get_pairwise_mat(embeddings[: pos_num, :], embeddings, self.use_similarity, self.squared_distances)
        indices_tuple = lmu.convert_to_pairs(indices_tuple, labels)
        return self.loss_method(mat, labels, indices_tuple)

    def pair_based_loss(self, mat, labels, indices_tuple):
        a1, p, a2, n = indices_tuple
        pos_pair, neg_pair = [], []
        if len(a1) > 0:
            pos_pair = mat[a1, p]
        if len(a2) > 0:
            neg_pair = mat[a2, n]

        return self._compute_loss(pos_pair, neg_pair, (a1, p, a2, n))

    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        a1, p, a2, n = indices_tuple
        if len(a1) > 0 and len(a2) > 0:
            pos_pairs = pos_pairs.unsqueeze(1) / self.temperature
            neg_pairs = neg_pairs / self.temperature
            n_per_p = (a2.unsqueeze(0) == a1.unsqueeze(1)).float()
            neg_pairs = neg_pairs*n_per_p
            neg_pairs[n_per_p==0] = float('-inf')
            max_val = torch.max(pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0].half())
            numerator = torch.exp(pos_pairs - max_val).squeeze(1)
            denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1) + numerator
            log_exp = torch.log((numerator/denominator) + 1e-20)
            return torch.mean(-log_exp)
        return 0

class NTXentLoss_wiki(GenericPairLoss):

    def __init__(self, temperature, **kwargs):
        super().__init__(**kwargs, use_similarity=True, mat_based_loss=False)
        self.temperature = temperature

    def forward(self, query_re_output, start_re_output, indices_tuple=None, attention=None):
        if self.normalize_embeddings:
            query_re_output = torch.nn.functional.normalize(query_re_output, p=2, dim=1)
            start_re_output = torch.nn.functional.normalize(start_re_output, p=2, dim=1)

        loss = self.compute_loss(query_re_output, start_re_output, indices_tuple)
        if loss == 0:
            loss = torch.sum(embeddings*0)
        return loss

    def compute_loss(self, query_re_output, start_re_output, indices_tuple):
        mat = lmu.get_pairwise_mat(query_re_output, start_re_output, self.use_similarity, self.squared_distances)
        return self.loss_method(mat, indices_tuple)

    def pair_based_loss(self, mat, indices_tuple):
        a1, p, a2, n = indices_tuple
        pos_pair, neg_pair = [], []
        if len(a1) > 0:
            pos_pair = mat[a1, p]
        if len(a2) > 0:
            neg_pair = mat[a2, n]

        return self._compute_loss(pos_pair, neg_pair, (a1, p, a2, n))

    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        a1, _, a2, _ = indices_tuple

        if len(a1) > 0 and len(a2) > 0:
            pos_pairs = pos_pairs.unsqueeze(1) / self.temperature
            neg_pairs = neg_pairs / self.temperature
            n_per_p = (a2.unsqueeze(0) == a1.unsqueeze(1)).float()
            neg_pairs = neg_pairs*n_per_p
            neg_pairs[n_per_p==0] = float('-inf')
            max_val = torch.max(pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0].half())
            numerator = torch.exp(pos_pairs - max_val).squeeze(1)
            denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1) + numerator
            log_exp = torch.log((numerator/denominator) + 1e-20)
            return torch.mean(-log_exp)
        return 0

class CP_R(nn.Module):
    def __init__(self, args):
        super(CP_R, self).__init__()
        if args.bert_model == 'bert':
            if os.path.exists('***path_to_your_bert_tokenizer***'):
                if args.cased == 0:
                    load_path = '***path_to_your_bert_tokenizer_uncased***'
                elif args.cased == 1:
                    load_path = '***path_to_your_bert_tokenizer_cased***'
            else:
                if args.cased == 0:
                    load_path = '***path_to_your_bert_tokenizer_uncased***'
                elif args.cased == 1:
                    load_path = '***path_to_your_bert_tokenizer_cased***'
            self.model = BertForMaskedLM.from_pretrained(load_path)
            self.tokenizer = BertTokenizer.from_pretrained(load_path)
        elif args.bert_model == 'roberta':
            if os.path.exists('***path_to_your_roberta_tokenizer***'):
                load_path = '***path_to_your_roberta_tokenizer***'
            else:
                load_path = '***path_to_your_roberta_tokenizer***'
            self.model = RobertaForMaskedLM.from_pretrained(load_path)
            self.tokenizer = RobertaTokenizer.from_pretrained(load_path)

        self.args = args
        self.ntxloss_doc = NTXentLoss_doc(temperature=args.temperature)
        self.ntxloss_wiki = NTXentLoss_wiki(temperature=args.temperature)

    def get_doc_loss(self, context_idxs, h_mapping, t_mapping, relation_label, relation_label_idx, context_masks, rel_mask_pos, rel_mask_neg, pos_num, mlm_mask):
        if self.args.doc_loss:
            m_input, m_labels = mask_tokens(context_idxs.cpu(), self.tokenizer, mlm_mask.cpu())
            m_outputs = self.model(input_ids=m_input, masked_lm_labels=m_labels, attention_mask=context_masks)
            m_loss = m_outputs[1]

            context_output = m_outputs[0]
            start_re_output = torch.matmul(h_mapping, context_output)
            end_re_output = torch.matmul(t_mapping, context_output)
            hidden = torch.cat([start_re_output, end_re_output], dim = 2)

            pair_hidden = []
            for i in range(relation_label_idx.size()[0]):
                pair_hidden.append(hidden[relation_label_idx[i][0], relation_label_idx[i][1]])
            pair_hidden = torch.stack(pair_hidden, dim = 0)

            def get_all_pairs_indices(labels, rel_mask_pos, rel_mask_neg):
                ref_labels = labels
                labels1 = labels.unsqueeze(1)
                labels2 = ref_labels.unsqueeze(0)
                matches = (labels1 == labels2).byte()
                diffs = matches ^ 1
                matches = matches * rel_mask_pos
                diffs = diffs * rel_mask_neg
                a1_idx = matches.nonzero()[:, 0].flatten()
                p_idx = matches.nonzero()[:, 1].flatten()
                a2_idx = diffs.nonzero()[:, 0].flatten()
                n_idx = diffs.nonzero()[:, 1].flatten()
                return a1_idx, p_idx, a2_idx, n_idx

            indices_tuple = get_all_pairs_indices(relation_label, rel_mask_pos, rel_mask_neg)

            r_loss = self.ntxloss_doc(pair_hidden, relation_label, indices_tuple, pos_num)
        else:
            m_loss = 0
            r_loss = 0

        return m_loss, r_loss

    def get_wiki_loss(self, context_idxs, h_mapping, query_mapping, context_masks, rel_mask_pos, rel_mask_neg, relation_label_idx, start_positions, end_positions, mlm_mask):
        m_input, m_labels = mask_tokens(context_idxs.cpu(), self.tokenizer, mlm_mask.cpu())
        m_outputs = self.model(input_ids=m_input, masked_lm_labels=m_labels, attention_mask=context_masks)
        m_loss = m_outputs[1]
        context_output = m_outputs[0]

        if self.args.wiki_loss == 1:
            query_re_output = torch.matmul(query_mapping.unsqueeze(dim = 1), context_output)
            query_re_output = query_re_output.squeeze(dim = 1)
            start_re_output = torch.matmul(h_mapping, context_output)
            new_start_re_output = []
            for i in range(relation_label_idx.size()[0]):
                new_start_re_output.append(start_re_output[relation_label_idx[i][0], relation_label_idx[i][1]])
            start_re_output = torch.stack(new_start_re_output, dim = 0)
            query_re_output = query_re_output

            def get_all_pairs_indices(rel_mask_pos, rel_mask_neg):
                a1_idx = rel_mask_pos.nonzero()[:, 0].flatten()
                p_idx = rel_mask_pos.nonzero()[:, 1].flatten()
                a2_idx = rel_mask_neg.nonzero()[:, 0].flatten()
                n_idx = rel_mask_neg.nonzero()[:, 1].flatten()
                return a1_idx, p_idx, a2_idx, n_idx

            indices_tuple = get_all_pairs_indices(rel_mask_pos, rel_mask_neg)
            r_loss = self.ntxloss_wiki(query_re_output, start_re_output, indices_tuple)
        else:
            r_loss = 0
        return m_loss, r_loss

    def forward(self, batch, doc_loss = 0, wiki_loss = 0):
        if doc_loss + wiki_loss != 1:
            assert False
        if doc_loss == 1:
            m_loss_d, r_loss_d = self.get_doc_loss(**batch[0])
            return m_loss_d, r_loss_d
        elif wiki_loss == 1:
            m_loss_w, r_loss_w = self.get_wiki_loss(**batch[1])
            return m_loss_w, r_loss_w
