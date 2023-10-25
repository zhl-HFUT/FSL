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
        self.model, self.para_model = prepare_model(args)
        for param, param_k in zip(self.model.encoder.parameters(), self.model.encoder_target.parameters()):
            param_k.data.copy_(param.data)
            param_k.requires_grad = False
        
        for param, param_k in zip(self.model.proj_head.parameters(), self.model.proj_head_target.parameters()):
            param_k.data.copy_(param.data)
            param_k.requires_grad = False
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)
        self.pass_ids = bool(args.pass_ids) # to remove same class instances during training using base dataset
        self.mixed_precision = args.mixed_precision
        if self.mixed_precision:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                opt_level=self.mixed_precision)
        
        if args.update_base_embeds:
            batch_size = 32
            self.update_loader = get_update_loader(args, batch_size)



    def prepare_label(self):
        args = self.args

        # prepare one-hot label
        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
        
        if args.model_class == 'FEATBaseTransformer3_Aux':
            label_aux = torch.arange(args.way, dtype=torch.int8).repeat(args.k-1)
        else:
            if args.label_aux_type is None:
                label_aux = torch.arange(args.way, dtype=torch.int8).repeat(args.shot + args.query)
            elif args.label_aux_type == 'random':
                label_aux = torch.randperm(args.way, dtype=torch.int8).repeat(args.shot + args.query)

            if args.simclr_loss_type=='ver2.1':
                label_aux = get_label_aux((args.shot+args.query)*(args.way))

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

    def train(self):
        args = self.args
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()
        elif self.args.fix_BN_only:
            self._fix_BN_only(self.model)
        
        # start FSL training
        label, label_aux = self.prepare_label()

        # print('before ver2.2 ', label_aux)

        print('Using mixed precision with opt level = ', self.mixed_precision)
        # asd

            

        for epoch in range(1, args.max_epoch + 1):
            self.train_epoch += 1
            self.model.train()
            if self.args.fix_BN:
                self.model.encoder.eval()
            elif self.args.fix_BN_only:
                self._fix_BN_only(self.model)
            
            tl1 = Averager()
            tl2 = Averager()
            ta = Averager()
            if self.args.dataset == 'MiniImageNet':
                train_classes = 64
            elif self.args.dataset == 'CUB':
                train_classes = 100
            elif self.args.dataset == 'TieredImageNet' or self.args.dataset == 'TieredImageNet_og':
                train_classes = 351
            tca = AccuracyClassAverager(train_classes)

            start_tm = time.time()

            if self.args.update_base_interval is not None:
                if epoch%self.args.update_base_interval==0:
                    print('running base proto update')
                    self.model.update_base_protos()
            
            if self.args.update_base_embeds:
                if self.trlog['max_acc_epoch']==epoch-1 and epoch>=args.patience:
                    self.model.update_2d_embeds(self.update_loader)

            for batch in self.train_loader:
                self.optimizer.zero_grad()


                self.train_step += 1

                data, gt_label, ids = batch[0].cuda(), batch[1].cuda(), batch[2]
                
                data_tm = time.time()
                self.dt.add(data_tm - start_tm)
                
                if self.args.method == 'proto_net_only':
                    logits = self.model(data, ids, key_cls=gt_label[:5])
                    loss_meta = F.cross_entropy(logits, label)
                    total_loss = loss_meta
                elif self.args.method == 'proto_FGKVM':
                    logits, kvmpred, kvmtarget = self.model(data, ids, key_cls=gt_label[:5])
                    loss_meta = F.cross_entropy(logits, label)
                    # total_loss = loss_meta + Lmempred
                    Lmempred = torch.tensor(80.).cuda()
                    # if self.train_step%100==0:
                    #     for pred in kvmpred:
                    #         for target in kvmtarget:
                    #             print(torch.nn.functional.cosine_similarity(pred, target, dim=0))
                    # for pred, target in zip(kvmpred, kvmtarget):
                    #     Lmempred -= torch.nn.functional.cosine_similarity(pred, target, dim=0)
                    #     if self.train_step%100==0:
                    #         print(torch.nn.functional.cosine_similarity(pred, target, dim=0))
                    # print('Lmempred:', Lmempred)
                    # total_loss = Lmempred + loss_meta*30
                    total_loss = loss_meta
                else:
                    logits, reg_logits, metrics, sims, pure_index = self.model(data, ids, key_cls=gt_label[:5])

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
                    total_loss = loss_meta# + loss_infoNCE_neg
                
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
                self.model._momentum_update_key_encoder()


                # step optimizer
                # self.optimizer.step()

                # if 'cycl' in self.args.lr_scheduler:
                #     # print('steppugn inside batch loop')
                #     self.lr_scheduler.step() 

                backward_tm = time.time()
                self.bt.add(backward_tm - forward_tm)
                
                self.optimizer.step() 


                optimizer_tm = time.time()
                self.ot.add(optimizer_tm - backward_tm)

                # refresh start_tm
                start_tm = time.time()
                
            if 'cycl' not in self.args.lr_scheduler:
                print('stepping outside batch loop')
                self.lr_scheduler.step()
      
            # print('logits range ', [logits.min().item(), logits.max().item()])
            # print('class wise test accuracies ')
            # print(tca.item())
            self.try_evaluate(epoch)
            self.save_model('epoch-last')
            if self.train_epoch%40 == 0:
                self.evaluate_test('epoch-last.pth')



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
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))

        # vca = AccuracyClassAverager(16)

        with torch.no_grad():
            for i, batch in enumerate(data_loader, 1):
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
        if self.args.fix_BN:
            self.model.encoder.eval()
        elif self.args.fix_BN_only:
            self._fix_BN_only(self.model)

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

    def evaluate_test(self, path):
        # restore model args
        args = self.args
        # evaluation mode
        if args.test:
            path = args.test
        else:
            path = osp.join(self.args.save_path, path)
        params = torch.load(path)['params']
        del params['queue_ptr']
        del params['queue']
        print('deleted')
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
            print('Epoch{} val acc={:.4f} + {:.4f}\n'.format(
                    epoch,
                    self.trlog['max_acc'],
                    self.trlog['max_acc_interval']))
        elif 'epoch-last.pth' in path:
            self.trlog['final_test_acc'] = va
            self.trlog['final_test_acc_interval'] = vap
            self.trlog['final_test_loss'] = vl
            epoch = self.args.max_epoch
            print('Epoch{} val acc={:.4f} + {:.4f}\n'.format(
                    epoch,
                    self.trlog['final_val_acc'],
                    self.trlog['final_val_acc_interval']))

        # print('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
        #         self.trlog['max_acc_epoch'],
        #         self.trlog['max_acc'],
        #         self.trlog['max_acc_interval']))
        print('Epoch{} Test acc={:.4f} + {:.4f}\n'.format(
                epoch,
                va,
                vap))

        # print('class wise test accuracies ')
        # print(tca.item())
        # if args.test:
        #     pkl_path = osp.dirname(args.test)
        # else:
        #     pkl_path = args.save_path
        
        # save_dict = tca.item()
        # save_dict['max_acc_epoch'] = self.trlog['max_acc_epoch']
        # save_dict['best_val_acc'] = self.trlog['max_acc']
        # save_dict['test_acc'] = self.trlog['test_acc']
        # json.dump(save_dict, open(osp.join(pkl_path, 'test_class_acc.json'), 'w'))
        # print('saved class accuracy json in {}'.format(pkl_path))

        # print('Saving class accuracy json in wandb as a file')
        # artifact = self.wandb_run.Artifact('test_accuracies', type='test_acc_json')
        # artifact.add_file(pkl_path)
        # self.wandb_run.log_artifact(artifact)
        # try_count = 0
        # while try_count<10:
        #     try:
        #         self.wandb_run.save(osp.join(pkl_path, 'test_class_acc.json'))
        #         break
        #     except Exception as ex:
        #         try_count = try_count + 1
        #         print([ex, 'Retrying'])
        #         time.sleep(10)
            
        # print('here    ')
        # print(save_dict)
        # self.wandb_run.log({"accuracies": wandb.Html(json2html.convert(json = save_dict))})
        # copyfile(osp.join(pkl_path, 'test_class_acc.json'),
        #     osp.join(self.wandb_run.dir, 'test_class_acc.json'))

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