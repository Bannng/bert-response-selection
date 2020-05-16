__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

from typing import Any

from transformers import BertForNextSentencePrediction
from utils.cuda_setting import get_device_setting


import torch.nn.functional as F
import torch.nn as nn


class NextSentencePrediction(nn.Module):
    def __init__(self, bert: BertForNextSentencePrediction) -> None:
        super().__init__()
        self.bert = bert

    def forward(self, eval, input_ids, segment_ids, attn_masks, labels) -> Any:
        loss, seq_rel_scores = self.bert(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            attention_mask=attn_masks,
            next_sentence_label=labels,
        )
        return loss, F.softmax(seq_rel_scores, dim=-1)