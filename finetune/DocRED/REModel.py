import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
import math

class REModel(nn.Module):
    def __init__(self, config, bert_model):
        super(REModel, self).__init__()
        self.config = config
        self.use_distance = True

        hidden_size = 128
        self.bert = bert_model

        bert_hidden_size = self.bert.config.hidden_size
        self.linear_re = nn.Linear(bert_hidden_size, hidden_size)
 
        if self.use_distance:
            self.dis_embed = nn.Embedding(20, config.dis_size, padding_idx=10)
            self.bili = torch.nn.Bilinear(hidden_size+config.dis_size, hidden_size+config.dis_size, config.relation_num)
        else:
            self.bili = torch.nn.Bilinear(hidden_size, hidden_size, config.relation_num)

    def forward(self, context_idxs, h_mapping, t_mapping, dis_h_2_t, dis_t_2_h, context_masks):
        context_output = self.bert(context_idxs, attention_mask=context_masks)[0]
        context_output = self.linear_re(context_output)
        start_re_output = torch.matmul(h_mapping, context_output)
        end_re_output = torch.matmul(t_mapping, context_output)

        if self.use_distance:
            s_rep = torch.cat([start_re_output, self.dis_embed(dis_h_2_t)], dim=-1)
            t_rep = torch.cat([end_re_output, self.dis_embed(dis_t_2_h)], dim=-1)
            predict_re = self.bili(s_rep, t_rep)
        else:
            predict_re = self.bili(start_re_output, end_re_output)

        return predict_re