__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


from typing import Any

import torch
from transformers import BertModel

import torch.nn as nn


class BertBaseCLS(nn.Module):
    def __init__(self, bert: BertModel) -> None:
        super().__init__()
        self.bert = bert
        self.cls = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(self.bert.config.hidden_size, int(self.bert.config.hidden_size / 2)),
            nn.SELU(),
            nn.Linear(int(self.bert.config.hidden_size / 2), 1),
            nn.Sigmoid()
        )

    def forward(self, eval, input_ids, segment_ids, attn_masks) -> torch.Tensor:
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

            # [(bs * 10) x hidden]
            pooled_output = self.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=attn_masks)[1]

            # [(bs * 10) x 1]
            logits = self.cls(pooled_output)
            logits = logits.squeeze(dim=-1)
            print('viewed logits shape', logits.shape)

            return logits
        else:
            """
            :input_ids [bs x max_len]
            :segment_ids [bs x max_len]
            :attn_masks [bs x max_len]
            :labels [bs]
            :returns loss, predictions
            """
            # [bs x hidden]
            pooled_output = self.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=attn_masks)[1]

            # [bs x 1]
            logits = self.cls(pooled_output).squeeze(dim=-1)

            return logits
