__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


from transformers import BertTokenizer
from typing import Any, Tuple


def get_tokenizer(model_name: str='bert-base-cased') -> Tuple[BertTokenizer, int]:
    eou_token, eot_token = '__eou__', '__eot__'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    special_tokens_dict = {'additional_special_tokens': [eou_token,  eot_token]}

    orig_num_tokens = len(tokenizer)
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    return tokenizer, orig_num_tokens + num_added_tokens
