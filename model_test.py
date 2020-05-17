__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

import torch
import torch.nn.functional as F
from transformers import BertForNextSentencePrediction, BertTokenizer
from models import response_selection


model_name = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
seq1 = "Hello , How are you ?"
seq2 = "I am fine , thank you ."
preprocessed = tokenizer.encode_plus(seq1, seq2, add_special_tokens=True, max_length=32, pad_to_max_length=True)

input_ids = torch.LongTensor([preprocessed['input_ids'], preprocessed['input_ids']])
attn_masks = torch.LongTensor([preprocessed['attention_mask'], preprocessed['attention_mask']])
segment_ids = torch.LongTensor([preprocessed['token_type_ids'], preprocessed['token_type_ids']])

model = BertForNextSentencePrediction.from_pretrained(model_name)

print(preprocessed)

output = model(input_ids=input_ids, attention_mask=attn_masks, token_type_ids=segment_ids, next_sentence_label=torch.LongTensor([0, 0]))
print(output)
print(F.softmax(output[1], dim=-1).max(0))