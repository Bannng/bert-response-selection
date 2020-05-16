__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


import torch
import pandas as pd


from typing import Iterable, Tuple, Any
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils.udc_data_util import get_tokenizer


class SiameseDialogDataset(Dataset):
    def __init__(self, ctx, utter, label, tokenizer, ctx_max_len, utter_max_len, eval=False) -> None:
        super().__init__()
        self.ctx = ctx
        self.utter = utter
        self.label = label
        self.tokenizer = tokenizer
        self.ctx_max_len = ctx_max_len
        self.utter_max_len = utter_max_len
        self.eval = eval

    def __getitem__(self, idx) -> Any:
        if self.eval == False:
            print(self.ctx[idx][:self.ctx_max_len-2])
            print(self.utter[idx])

            if len(self.ctx[idx]) < self.ctx_max_len:
                composition = self.ctx[idx] + ' [SEP] ' + self.utter[idx]
                composition = self.tokenizer.encode_plus(
                    composition[:self.ctx_max_len + self.utter_max_len],
                    add_special_tokens=True,
                    pad_to_max_length=True,
                    max_length=self.ctx_max_len + self.utter_max_len
                )
            else:
                composition = self.ctx[idx][:self.ctx_max_len-2] + ' [SEP] ' + self.utter[idx]
                composition = self.tokenizer.encode_plus(
                    composition[:self.ctx_max_len + self.utter_max_len],
                    add_special_tokens=True,
                    pad_to_max_length=True,
                    max_length=self.ctx_max_len + self.utter_max_len
                )

            input_ids = torch.LongTensor(composition['input_ids'])
            segment_ids = torch.LongTensor(composition['token_type_ids'])
            attn_masks = torch.LongTensor(composition['attention_mask'])

            return input_ids, segment_ids, attn_masks, self.label[idx]
        elif self.eval == True:
            composition = []

            for u in self.utter[idx]:
                if len(self.ctx[idx]) < self.ctx_max_len:
                    composition.append(self.ctx[idx] + ' [SEP] ' + u)
                else:
                    composition.append(self.ctx[idx][:self.ctx_max_len - 2] + ' [SEP] ' + u)

            composition = [self.tokenizer.encode_plus(c,
                                                      add_special_tokens=True,
                                                      pad_to_max_length=True,
                                                      max_length=self.ctx_max_len + self.utter_max_len)
                           for c in composition]
            input_ids = [torch.LongTensor(c['input_ids']) for c in composition]
            segment_ids = [torch.LongTensor(c['token_type_ids']) for c in composition]
            attn_masks = [torch.LongTensor(c['attention_mask']) for c in composition]
            labels = [1] + [0] * 9

            return input_ids, segment_ids, attn_masks, torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.utter)


if __name__ == '__main__':
    valid = pd.read_csv('../rsc/data/valid.csv')
    valid['combined'] = valid.apply(lambda x: list([x['Ground Truth Utterance'], x['Distractor_0'], x['Distractor_1'], x['Distractor_2'], x['Distractor_3'], x['Distractor_4'], x['Distractor_5'], x['Distractor_6'], x['Distractor_7'], x['Distractor_8']]), axis=1)
    distractors = valid['combined'].values.tolist()[:5]
    contexts = valid['Context'].values.tolist()[:5]

    tokenizer, num_tokens = get_tokenizer()
    print(tokenizer, num_tokens)

    loader = DataLoader(SiameseDialogDataset(contexts, distractors, None, tokenizer, 128, 64, True), batch_size=2, shuffle=False)
    print(len(loader))

    for i, (input_ids, segment_ids, attn_masks, labels) in enumerate(loader):
        print(input_ids)
        break



