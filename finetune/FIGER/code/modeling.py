import os
import pdb
import torch
import torch.nn as nn 
from transformers import BertForMaskedLM, BertTokenizer, BertForPreTraining
import random
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil
from torch.nn import functional as F

class BertForEntityTyping(nn.Module):
    def __init__(self, bert, num_labels=2):
        super(BertForEntityTyping, self).__init__()
        self.bert = bert
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.1)
        self.typing = nn.Linear(768, num_labels, False)
        for layer in [self.typing]:
            torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, input_ids, bert_model, token_type_ids=None, attention_mask=None, span_mask=None, labels=None):
        if bert_model == 'bert':
            context, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        elif bert_model == 'roberta':
            context, pooled_output = self.bert(input_ids, attention_mask=attention_mask)
        pooled_context = self.dropout(context)
        pooled_context = torch.matmul(span_mask.unsqueeze(dim = 1), context)

        logits = self.typing(pooled_context).squeeze(dim = 1)
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss
        else:
            return logits
