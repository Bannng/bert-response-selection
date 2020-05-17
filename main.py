__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


from torch.utils.data import DataLoader
from transformers import BertForNextSentencePrediction
from models.response_selection import NextSentencePrediction
from trainer import Trainer
from utils.udc_data_loader import SiameseDialogDataset
from utils.udc_data_util import get_tokenizer


import pandas as pd
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', default=5, type=int, required=True)
    parser.add_argument('--train_bs', default=32, type=int, required=True)
    parser.add_argument('--valid_bs', default=16, type=int, required=True)
    parser.add_argument('--ctx_max_len', default=128, type=int, required=True)
    parser.add_argument('--utter_max_len', default=64, type=int, required=True)
    parser.add_argument('--lr', default=1e-5, type=float, required=True)
    parser.add_argument('--eps', default=1e-5, type=float, required=True)
    parser.add_argument('--log_dir', default='./runs/', type=str)
    parser.add_argument('--num_workers', default=8, type=int)

    parse = parser.parse_args()

    train = pd.read_csv('./rsc/data/train.csv')
    train_utter, train_ctx, train_label = train['Utterance'], train['Context'], train['Label']

    tokenizer, num_tokens = get_tokenizer()

    train_loader = DataLoader(
        SiameseDialogDataset(train_ctx, train_utter, train_label, tokenizer, parse.ctx_max_len,
                             parse.utter_max_len, False),
        batch_size=parse.train_bs,
        shuffle=False,
        pin_memory=True,
        num_workers=parse.num_workers
    )

    bert = BertForNextSentencePrediction.from_pretrained('bert-base-cased')
    bert.resize_token_embeddings(num_tokens)

    model = NextSentencePrediction(bert)

    valid = pd.read_csv('./rsc/data/valid.csv')
    valid['combined'] = valid.apply(lambda x: list(
        [x['Ground Truth Utterance'], x['Distractor_0'], x['Distractor_1'], x['Distractor_2'], x['Distractor_3'],
         x['Distractor_4'], x['Distractor_5'], x['Distractor_6'], x['Distractor_7'], x['Distractor_8']]), axis=1)
    distractors = valid['combined'].values.tolist()
    contexts = valid['Context'].values.tolist()

    tokenizer, num_tokens = get_tokenizer()

    valid_loader = DataLoader(
        SiameseDialogDataset(contexts, distractors, None, tokenizer, parse.ctx_max_len, parse.utter_max_len, True),
        batch_size=16, shuffle=False, pin_memory=True, num_workers=parse.num_workers)

    args = {
        'lr': parse.lr,
        'eps': parse.eps,
        'epoch': parse.epoch,
        'log_dir': parse.log_dir
    }

    Trainer(model, args).train(train_loader, valid_loader)

