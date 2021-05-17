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

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, input_ent=None, ent_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, input_ent, ent_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.typing(pooled_output)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss
        else:
            return logits