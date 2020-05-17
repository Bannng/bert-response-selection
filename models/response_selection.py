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
        if eval:
            """
            :input_ids [bs x 10 x max_len]
            :segment_ids [bs x 10 x max_len]
            :attn_masks [bs x 10 x max_len]
            :labels [bs x 10]
            :returns loss, predictions [bs x 10]
            """
            bs, max_len = input_ids.size(0), input_ids.size(2)
            seq_count = int(bs * 10)

            # [(bs * 10) x max_len]
            input_ids = input_ids.view(seq_count, max_len)
            segment_ids = segment_ids.view(seq_count, max_len)
            attn_masks = attn_masks.view(seq_count, max_len)
            # [seq_count]
            labels = labels.view(seq_count)

            loss, seq_rel_scores = self.bert(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=attn_masks,
                next_sentence_label=labels,
            )

            preds = F.softmax(seq_rel_scores, dim=-1).max(dim=1)[1].view(bs, 10).tolist()
            return loss, preds
        else:
            """
            :input_ids [bs x max_len]
            :segment_ids [bs x max_len]
            :attn_masks [bs x max_len]
            :labels [bs]
            :returns loss, predictions
            """

            labels = labels.squeeze(dim=-1)
            loss, seq_rel_scores = self.bert(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=attn_masks,
                next_sentence_label=labels,
            )

            preds = F.softmax(seq_rel_scores, dim=-1).max(dim=1)[1]

            return loss, (labels == preds).sum().item()