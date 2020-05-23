__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from metric import calculate_candidates_ranking, logits_recall_at_k
from models.bert_base_cls import BertBaseCLS
from utils.cuda_setting import get_device_setting


class Trainer(object):

    def __init__(self, model: BertBaseCLS, args: dict) -> None:
        """
        Initialize Trainer Class
        :param model: BertForNextSentencePrediction
        :param args: {lr, eps, epoch, log_dir, ...}
        """

        self.model = model
        self.model = nn.DataParallel(self.model)
        self.model = self.model.to(get_device_setting())
        self.args = args
        self.writer = SummaryWriter(log_dir=args['log_dir'])
        self.optim = AdamW(self.model.parameters(), lr=args['lr'], eps=args['eps'])
        self.loss_fn = nn.BCEWithLogitsLoss().to(get_device_setting())
        # self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

    def train(self, train_loader: DataLoader, valid_loader: DataLoader) -> None:
        self.model.train()
        global_step = 0
        max_grad_norm = 2.0

        for ep in tqdm(range(1, self.args['epoch'] + 1)):
            for i, (input_ids, segment_ids, attn_masks, labels) in tqdm(enumerate(train_loader)):
                global_step += 1
                self.optim.zero_grad()
                logits = self.model(False,
                                    input_ids.to(get_device_setting()),
                                    segment_ids.to(get_device_setting()),
                                    attn_masks.to(get_device_setting()))

                labels = labels.squeeze(-1).float().to(get_device_setting())

                loss = self.loss_fn(logits, labels)
                loss = loss.mean()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optim.step()
                
                if global_step % 100 == 0:
#                     print(f'************** Training Total Loss : {loss.item()} ******************')
                    self.writer.add_scalar('Train/Total_Loss', loss.item(), global_step)

                if global_step % 10000 == 0:
                    self.evaluate(valid_loader, 'Valid', global_step)
                    torch.save(self.model.state_dict(), f'./rsc/output/bert-cls-v3-{global_step}.pth')

    def evaluate(self, valid_loader: DataLoader, mode: str, gs: int):
        self.model.eval()
        k = [1, 2, 3, 5, 7]
        total_examples, total_correct = 0, 0

        with torch.no_grad():
            for idx, (input_ids, segment_ids, attn_masks, labels) in tqdm(enumerate(valid_loader)):
                logits = self.model(True,
                                    input_ids.to(get_device_setting()),
                                    segment_ids.to(get_device_setting()),
                                    attn_masks.to(get_device_setting()))
                pred = torch.sigmoid(logits)
                pred = pred.cpu().detach().tolist()
                labels = labels.view(len(pred))
                labels = labels.cpu().detach().tolist()

                rank_by_pred = calculate_candidates_ranking(np.array(pred),
                                                            np.array(labels))

                num_correct, pos_index = logits_recall_at_k(rank_by_pred, k)
                total_correct = np.add(total_correct, num_correct)
                total_examples += rank_by_pred.shape[0]

                recall_result = ""
                
                if (idx + 1) % 1000 == 0:
                    for i in range(len(k)):
                        recall_result += "Recall@%s : " % k[i] + "%.2f%% | " % (float((total_correct[i]) / float(total_examples)) * 100)

                    print(recall_result)
                    
            for i in range(len(k)):
                print(f'Recall@{k[i]} -> {float(total_correct[i]) / float(total_examples)}')
                self.writer.add_scalar(f'{mode}/Recall@{k[i]}', (float(total_correct[i]) / float(total_examples)), gs)

        self.model.train()
