import time
import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F

from model.trainer.base import Trainer
from model.trainer.helpers import (
    get_dataloader, prepare_model, prepare_optimizer, get_update_loader
)
from model.utils import (
    pprint, ensure_path,
    Averager, Timer, count_acc, one_hot,
    compute_confidence_interval,
    AccuracyClassAverager
)
from tensorboardX import SummaryWriter
import wandb
from collections import deque
from tqdm import tqdm
import json
from shutil import copyfile
from json2html import *
from apex import amp
def get_label_aux(n_batch):
    label_aux = torch.cat([torch.arange(start=n_batch-2, end=2*n_batch-2),
        torch.arange(start=0, end=n_batch)])

    return label_aux

from apex import amp

class FSLTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)


        self.return_simclr = True if args.return_simclr is not None else False
        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.model = prepare_model(args)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)
        self.pass_ids = bool(args.pass_ids) # to remove same class instances during training using base dataset
        self.mixed_precision = args.mixed_precision
        if self.mixed_precision:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                opt_level=self.mixed_precision)



    def prepare_label(self):
        args = self.args

        # prepare one-hot label
        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)

        label_aux = torch.arange(args.way, dtype=torch.int8).repeat(args.shot + args.query)

        label = label.type(torch.LongTensor)
        label_aux = label_aux.type(torch.LongTensor)
        # print('label_aux ', label_aux)
        if torch.cuda.is_available():
            label = label.cuda()
            label_aux = label_aux.cuda()
            
        return label, label_aux
    
    
    def _fix_BN_only(self, model):
        for module in model.encoder.modules():
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                module.eval()


    def _fix_BN_only(self, model):
        for module in model.encoder.modules():
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                module.eval()

    def train(self):
        args = self.args
        
        label, label_aux = self.prepare_label()

        print('Using mixed precision with opt level = ', self.mixed_precision)

        for epoch in range(1, args.max_epoch + 1):
            self.train_epoch += 1
            self.model.train()
            
            tl1 = Averager()
            tl2 = Averager()
            ta = Averager()

            start_tm = time.time()

            for batch in self.train_loader:
                self.optimizer.zero_grad()


                self.train_step += 1

                # data, gt_label, ids = batch[0].cuda(), batch[1].cuda(), batch[2]
                if self.return_simclr:
                    data, gt_label, ids, data_simclr = batch[0].cuda(), batch[1].cuda(), batch[2], batch[3].cuda()
                else:
                    data, gt_label, ids = batch[0].cuda(), batch[1].cuda(), batch[2]
                    data_simclr = None
                
                data_tm = time.time()
                self.dt.add(data_tm - start_tm)
                
                logits, logits_simclr, metrics, sims, pure_index = self.model(data, ids, simclr_images=data_simclr, key_cls=gt_label[:5])

                sims = torch.tensor(sims).cuda()
                pure_index = torch.tensor(pure_index).cuda()
                pos_index = []
                for j in range(len(sims)):
                    if sims[j] >= 0.8:
                        pos_index.append(j)
                pos_index = torch.tensor(pos_index).cuda()
                weight_sum = sims.sum()
                metric_exp_sum = torch.exp(metrics).sum()
                label_moco = torch.tensor(0).type(torch.cuda.LongTensor)
                # loss_infoNCE = loss_infoNCE + F.cross_entropy(metrics, label_moco)
                loss_infoNCE_neg = F.cross_entropy(torch.index_select(metrics, 0, pure_index).unsqueeze(0), label_moco.unsqueeze(0))
                # loss_sup_con = - (torch.log(torch.exp(torch.index_select(metrics, 0, pos_index)) / metric_exp_sum) * torch.index_select(sims, 0, pos_index)).sum()/weight_sum
                # print('loss_sup_con:', loss_sup_con.item())
                # loss_pos = torch.tensor(1.0/0.07).cuda() - metrics[0]

                # loss = self.loss(logits, label)
                # total_loss = self.loss(logits, label)
                loss_meta = F.cross_entropy(logits, label)

                aux_loss = 0
                if args.balance > 0:
                    aux_loss = F.cross_entropy(logits_simclr, self.model.label_aux)
                # print(aux_loss)

                total_loss = loss_meta + loss_infoNCE_neg + args.balance * aux_loss
                
                tl2.add(loss_meta)
                forward_tm = time.time()
                self.ft.add(forward_tm - data_tm)
                acc = count_acc(logits, label)

                tl1.add(total_loss.item())
                ta.add(acc)
                self.try_logging(total_loss, loss_meta, acc, aux_loss=None)
                # tca.add(logits.cpu(), gt_label[self.args.way:].cpu())

                # self.optimizer.zero_grad()
                # total_loss.backward() # unhide this for non mixed precision
                if self.mixed_precision:
                    with amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    total_loss.backward()

                backward_tm = time.time()
                self.bt.add(backward_tm - forward_tm)
                
                self.optimizer.step() 

                optimizer_tm = time.time()
                self.ot.add(optimizer_tm - backward_tm)

                # refresh start_tm
                start_tm = time.time()
                

            self.lr_scheduler.step()
      
            # print('logits range ', [logits.min().item(), logits.max().item()])
            # print('class wise test accuracies ')
            # print(tca.item())
            self.try_evaluate(epoch)
            self.save_model('epoch-last')
            if self.train_epoch%20 == 0:
                self.evaluate_test('epoch-last.pth', specified_epoch=self.train_epoch)

            print('ETA:{}/{}'.format(
                    self.timer.measure(),
                    self.timer.measure(self.train_epoch / args.max_epoch))
            )


        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        self.save_model('epoch-last')
        print('\n'+'start test'+'\n')
        

    def evaluate(self, data_loader):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.eval()
        record = np.zeros((args.num_eval_episodes, 2)) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best test600 acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))

        # vca = AccuracyClassAverager(16)

        with torch.no_grad():
            for i, batch in enumerate(data_loader, 1):
                if i == 600:
                    break
                if torch.cuda.is_available():
                    if self.pass_ids:
                        data, gt_label, ids = batch[0].cuda(), batch[1].cuda(), batch[2]
                    else:
                        data, gt_label = [_.cuda() for _ in batch]
                else:
                    if self.pass_ids:
                        data, gt_label, ids = batch[0], batch[1], batch[2]
                    else:
                        data, gt_label = batch[0], batch[1]
                if self.pass_ids:
                    logits = self.model(data, ids)
                else:
                    logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc
                # vca.add(logits.cpu(), gt_label[self.args.way:].cpu())
                
        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
        # print('class wise test accuracies ')
        # print(vca.item())
        # train mode
        self.model.train()

        return vl, va, vap

    def load_model(self):
        args = self.args
        path = args.test
        print('setting state dict of model to {}'.format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['params'])
        if self.mixed_precision is not None:
            print('setting amp state dict ')
            amp.load_state_dict(checkpoint['amp'])

    def save_model(self, name):
        if self.mixed_precision is not None:
            save_dict = dict(params=self.model.state_dict(),
            amp=amp.state_dict())
        else:
            save_dict = dict(params=self.model.state_dict())

        torch.save(
            save_dict,
            osp.join(self.args.save_path, name + '.pth')
        )

    def evaluate_test(self, path, specified_epoch=None):
        # restore model args
        args = self.args
        # evaluation mode
        if args.test:
            path = args.test
        else:
            path = osp.join(self.args.save_path, path)
        params = torch.load(path)['params']
        # del params['queue_ptr']
        # del params['queue']
        # print('deleted')
        self.model.load_state_dict(params, strict=False)
        self.model.eval()
        record = np.zeros((10000, 2)) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        # print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
        #         self.trlog['max_acc_epoch'],
        #         self.trlog['max_acc'],
        #         self.trlog['max_acc_interval']))
        if self.args.dataset == 'MiniImageNet':
            test_classes = 20
        elif self.args.dataset == 'CUB':
            test_classes = 50
        elif self.args.dataset == 'TieredImageNet' or self.args.dataset == 'TieredImageNet_og':
            test_classes = 160
        tca = AccuracyClassAverager(test_classes)

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.test_loader, 1)):
                # if torch.cuda.is_available():
                #     data, gt_label = [_.cuda() for _ in batch]
                # else:
                #     data = batch[0]
                if torch.cuda.is_available():
                    if self.pass_ids:
                        data, gt_label, ids = batch[0].cuda(), batch[1].cuda(), batch[2]
                    else:
                        data, gt_label = [_.cuda() for _ in batch]
                else:
                    if self.pass_ids:
                        data, gt_label, ids = batch[0], batch[1], batch[2]
                    else:
                        data, gt_label = batch[0], batch[1]
                if self.pass_ids:
                    logits = self.model(data, ids)
                else:
                    logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc
                # print([data.shape, gt_label.shape])
                # print('here')
                # print(logits.cpu().shape)
                # print(gt_label[self.args.eval_shot*self.args.way:].cpu().shape)
                # asd
                tca.add(logits.cpu(), gt_label[self.args.eval_shot*self.args.way:].cpu())
       

     
        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
        
        if 'max_acc.pth' in path:
            self.trlog['test_acc'] = va
            self.trlog['test_acc_interval'] = vap
            self.trlog['test_loss'] = vl
            epoch = self.trlog['max_acc_epoch']
            print('Epoch{} test600 acc={:.4f} + {:.4f}\n'.format(
                    epoch,
                    self.trlog['max_acc'],
                    self.trlog['max_acc_interval']))
        elif 'epoch-last.pth' in path:
            self.trlog['final_test_acc'] = va
            self.trlog['final_test_acc_interval'] = vap
            self.trlog['final_test_loss'] = vl
            epoch = self.args.max_epoch
            if specified_epoch:
                epoch = specified_epoch
            print('Epoch{} test600 acc={:.4f} + {:.4f}\n'.format(
                    epoch,
                    self.trlog['final_val_acc'],
                    self.trlog['final_val_acc_interval']))

        # print('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
        #         self.trlog['max_acc_epoch'],
        #         self.trlog['max_acc'],
        #         self.trlog['max_acc_interval']))
        print('Epoch{} Test10000 acc={:.4f} + {:.4f}\n'.format(
                epoch,
                va,
                vap))

        return vl, va, vap
    
    def final_record(self):
        # save the best performance in a txt file
        
        with open(osp.join(self.args.save_path, 'epoch{}_{:.4f}+{:.4f}'.format(self.trlog['max_acc_epoch'], self.trlog['test_acc'], self.trlog['test_acc_interval'])), 'w') as f:
            f.write('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
            f.write('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))     
        with open(osp.join(self.args.save_path, 'epoch{}_{:.4f}+{:.4f}'.format(self.args.max_epoch, self.trlog['final_test_acc'], self.trlog['final_test_acc_interval'])), 'w') as f:
            f.write('final epoch {}, final val acc={:.4f} + {:.4f}\n'.format(
                self.args.max_epoch,
                self.trlog['final_val_acc'],
                self.trlog['final_val_acc_interval']))
            f.write('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['final_test_acc'],
                self.trlog['final_test_acc_interval']))    
        data = [[self.trlog['max_acc_epoch'], self.trlog['test_acc']], [self.args.max_epoch, self.trlog['final_test_acc']]]
        # table = wandb.Table(data=data, columns = ["epoch", "acc"])

        table = wandb.Table(data=data, columns = ["Epoch", "Acc"])
        self.wandb_run.log(
            {"Test Acc" : wandb.plot.line(table, "Epoch", "Acc",
                title="Test Acc")})
        # self.wandb_run.log({"Test Acc": table})