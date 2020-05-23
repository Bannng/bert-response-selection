__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

from utils.cuda_setting import get_device_setting


class Trainer(object):

    def __init__(self, model: nn.Module, args: dict) -> None:
        """
        Initialize Trainer Class
        :param model: BertForNextSentencePrediction
        :param args: {lr, eps, epoch, log_dir, ...}
        """

        self.model = model.to(get_device_setting())
        self.args = args
        self.writer = SummaryWriter(log_dir=args['log_dir'])
        self.optim = AdamW(self.model.parameters(), lr=args['lr'], eps=args['eps'])

    def train(self, train_loader: DataLoader, valid_loader: DataLoader) -> None:
        self.model.train()
        global_step = 0
        max_grad_norm = 2.0

        correct, bs = 0, 0

        for ep in tqdm(range(1, self.args['epoch'] + 1)):
            for i, (input_ids, segment_ids, attn_masks, labels) in tqdm(enumerate(train_loader)):
                global_step += 1
                self.optim.zero_grad()
                loss, preds = self.model(False,
                                         input_ids.to(get_device_setting()),
                                         segment_ids.to(get_device_setting()),
                                         attn_masks.to(get_device_setting()),
                                         labels.to(get_device_setting()))
                correct += preds
                bs += input_ids.size(0)
                acc = correct / bs

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optim.step()
                
                if global_step % 10 ==0:
                    self.evaluate(valid_loader, 'Valid', global_step)

                if global_step % 100 == 0:
                    print(f'************** Training Total Loss : {loss.item()} ******************')
                    print(f'************** Training Accuracy  : {acc} ******************')
                    self.writer.add_scalar('Train/Total_Loss', loss.item(), global_step)
                    self.writer.add_scalar('Train/Accuracy', acc, global_step)

                if global_step % 1000 == 0:
                    self.evaluate(valid_loader, 'Valid', global_step)
                    torch.save(self.model.state_dict(), f'./rsc/output/bert-nsp-{global_step}.pth')

    def evaluate(self, valid_loader: DataLoader, mode: str, gs: int):
        self.model.eval()
        total_count = 0
        recall_result = [0, 0, 0, 0, 0]  # 1, 2, 3, 5, 7
        k = [1, 2, 3, 5, 7]

        with torch.no_grad():
            for idx, (input_ids, segment_ids, attn_masks, labels) in tqdm(enumerate(valid_loader)):
                loss, preds, labels = self.model(True,
                                                 input_ids.to(get_device_setting()),
                                                 segment_ids.to(get_device_setting()),
                                                 attn_masks.to(get_device_setting()),
                                                 labels.to(get_device_setting()))

                bs = input_ids.size(0)

                total_count += bs
                recall = [self.evaluate_recall(preds, labels, k=i) for i in k]

                for i, r_k in enumerate(recall):
                    recall_result[i] += r_k

                if idx % 10 == 0:
                    for i, r_k in zip(k, recall_result):
                        print(f'Recall @ {i} : {r_k / float(total_count)}')

            for i, r_k in zip(k, recall_result):
                self.writer.add_scalar(f'{mode}/Recall@{i}', (r_k / float(total_count)), gs)

        self.model.train()

    def evaluate_recall(self, y, y_test, k=1):
        num_examples = float(len(y))
        num_correct = 0
        for predictions, label in zip(y, y_test):
            if label in predictions[:k]:
                num_correct += 1

        return num_correct