import abc
import torch

import numpy as np

import torch.nn.functional as F

from model.utils import count_acc, compute_confidence_interval
from tqdm import tqdm

from model.utils import Timer

class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args
        self.save_path = args.save_path
        self.train_step = 0
        self.train_epoch = 0
        self.max_steps = args.episodes_per_epoch * args.max_epoch
        self.timer = Timer()

        self.trlog = {}
        self.trlog['max_acc'] = 0.0
        self.trlog['max_acc_epoch'] = 0
        self.trlog['max_acc_interval'] = 0.0

    def prepare_label(self):
        args = self.args
        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
        label_aux = torch.arange(args.way, dtype=torch.int8).repeat(args.shot + args.query)
        label = label.type(torch.LongTensor).cuda()
        label_aux = label_aux.type(torch.LongTensor).cuda()
        return label, label_aux

    def logging(self, total_loss, loss_meta, loss_infoNCE_neg, acc):
        print('epoch {}, train {:06g}/{:06g}, loss={:.4f}, meta={:.4f}, NCE={:.4f} acc={:.4f}, lr={:.4g}'
                .format(self.train_epoch,
                        self.train_step,
                        self.max_steps,
                        total_loss.item(), loss_meta.item(), loss_infoNCE_neg.item(), acc,
                        self.optimizer.param_groups[0]['lr']))

    def save_model(self, name):
        import os.path as osp
        save_dict = dict(params=self.model.state_dict())
        torch.save(save_dict, osp.join(self.args.save_path, name + '.pth'))

    def load_model(self):
        args = self.args
        path = args.test
        print('setting state dict of model to {}'.format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['params'])

    def test(self, num_task):
        self.model.eval()
        label = torch.arange(self.args.way, dtype=torch.int16).repeat(self.args.query).type(torch.LongTensor).cuda()
        accs = []
        accs_blstm = []
        accs_mixs = [[] for _ in range(100)]
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader, 1):
                if i == num_task:
                    break
                data, gt_label, ids = batch[0].cuda(), batch[1].cuda(), batch[2]
                logits, logits_blstm = self.model(data, ids, key_cls=gt_label[:self.args.way])
                accs.append(count_acc(logits, label))
                if self.args.use_blstm_meta:
                    accs_blstm.append(count_acc(logits_blstm, label))
                    for i in range(100):
                        accs_mixs[i].append(count_acc(logits+logits_blstm*(i+1)*0.01, label))
        if self.args.use_blstm_meta:
            print('Test', num_task, f'acc = {np.mean(accs) * 100:.4f}')
            print('Test', num_task, f'blstm acc = {np.mean(accs_blstm) * 100:.4f}')
            for i in range(100):
                print((i+1)/100, f'mix acc = {np.mean(accs_mixs[i]) * 100:.4f}')
        else:
            print('Test', num_task, f'acc = {np.mean(accs) * 100:.4f}')
        self.model.train()
    
    def mini_test(self):
        self.model.eval()
        record = np.zeros((self.args.num_eval_episodes, 2)) # loss and acc
        label = torch.arange(self.args.way, dtype=torch.int16).repeat(self.args.query).type(torch.LongTensor).cuda()
        print('best mini_test epoch={}, acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))

        with torch.no_grad():
            for i, batch in enumerate(self.test_loader, 1):
                if i == 600:
                    break
                data, gt_label, ids = batch[0].cuda(), batch[1].cuda(), batch[2]
                logits = self.model(data, ids)
                record[i-1, 0] = F.cross_entropy(logits, label).item()
                record[i-1, 1] = count_acc(logits, label)

        loss = np.mean(record[:,0])
        acc, interval = compute_confidence_interval(record[:,1])
        
        print('epoch {}, mini_test, loss={:.4f} acc={:.4f}+{:.4f}'.format(
                    self.train_epoch, loss, acc, interval))

        if acc >= self.trlog['max_acc']:
            self.trlog['max_acc'] = acc
            self.trlog['max_acc_interval'] = interval
            self.trlog['max_acc_epoch'] = self.train_epoch
            self.save_model('max_acc')

        self.model.train()

    def test_100k(self):
        self.model.eval()
        accs = np.zeros(10000)
        label = torch.arange(self.args.way, dtype=torch.int16).repeat(self.args.query).type(torch.LongTensor).cuda()

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.test_loader, 1)):
                data, gt_label, ids = batch[0].cuda(), batch[1].cuda(), batch[2]
                logits = self.model(data, ids)
                accs[i-1] = count_acc(logits, label)

        acc, interval = compute_confidence_interval(accs)

        print('Epoch{} test_100k acc={:.4f} + {:.4f}\n'.format(self.train_epoch, acc, interval))