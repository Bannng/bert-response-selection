__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertForNextSentencePrediction
from models.response_selection import NextSentencePrediction
from trainer import Trainer
from utils.udc_data_loader import SiameseDialogDataset
from utils.udc_data_util import get_tokenizer

if __name__ == '__main__':
    train = pd.read_csv('./rsc/data/train.csv')
    train_utter, train_ctx, train_label = train['Utterance'], train['Context'], train['Label']

    tokenizer, num_tokens = get_tokenizer()

    train_loader = DataLoader(SiameseDialogDataset(train_ctx, train_utter, train_label, tokenizer, 128, 64, False), batch_size=32, shuffle=False)

    bert = BertForNextSentencePrediction.from_pretrained('bert-base-cased')
    bert.resize_token_embeddings(num_tokens)

    model = NextSentencePrediction(bert)

    args = {
        'lr': 1e-5,
        'eps': 1e-5,
        'epoch': 5,
        'log_dir': './runs/'
    }

    valid = pd.read_csv('./rsc/data/valid.csv')
    valid['combined'] = valid.apply(lambda x: list(
        [x['Ground Truth Utterance'], x['Distractor_0'], x['Distractor_1'], x['Distractor_2'], x['Distractor_3'],
         x['Distractor_4'], x['Distractor_5'], x['Distractor_6'], x['Distractor_7'], x['Distractor_8']]), axis=1)
    distractors = valid['combined'].values.tolist()
    contexts = valid['Context'].values.tolist()

    tokenizer, num_tokens = get_tokenizer()
    print(tokenizer, num_tokens)

    valid_loader = DataLoader(SiameseDialogDataset(contexts, distractors, None, tokenizer, 128, 64, True), batch_size=16,
                        shuffle=False)

    Trainer(model, args).train(train_loader, valid_loader)

