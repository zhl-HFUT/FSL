import torch
import torch.nn.functional as F

from model.trainer.base import Trainer
from model.trainer.helpers import (
    get_dataloader, prepare_model, prepare_optimizer
)
from model.utils import Averager, count_acc
from apex import amp

class FSLTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.return_simclr = True if args.return_simclr is not None else False
        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.model = prepare_model(args)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)
        self.mixed_precision = args.mixed_precision
        if self.mixed_precision:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                opt_level=self.mixed_precision)

    def train(self):
        args = self.args
        print('Using mixed precision with opt level = ', self.mixed_precision)
        
        label, label_aux = self.prepare_label()

        for epoch in range(1, args.max_epoch + 1):
            self.train_epoch += 1
            self.model.train()
            
            tl1 = Averager()
            tl2 = Averager()
            ta = Averager()

            for batch in self.train_loader:
                self.optimizer.zero_grad()
                self.train_step += 1

                if self.return_simclr:
                    data, gt_label, ids, data_simclr = batch[0].cuda(), batch[1].cuda(), batch[2], batch[3].cuda()
                else:
                    data, gt_label, ids = batch[0].cuda(), batch[1].cuda(), batch[2]
                    data_simclr = None
                
                logits, logits_simclr, metrics, sims, pure_index = self.model(data, 
                        ids, simclr_images=data_simclr, key_cls=gt_label[:args.way])
                loss_meta = F.cross_entropy(logits, label)
                total_loss = loss_meta

                if args.balance > 0:
                    aux_loss = F.cross_entropy(logits_simclr, self.model.label_aux)
                    total_loss += args.balance * aux_loss

                if args.use_infoNCE:
                    sims = torch.tensor(sims).cuda()
                    pure_index = torch.tensor(pure_index).cuda()
                    pos_index = []
                    for j in range(len(sims)):
                        if sims[j] >= 0.8:
                            pos_index.append(j)
                    pos_index = torch.tensor(pos_index).cuda()
                    # weight_sum = sims.sum()
                    # metric_exp_sum = torch.exp(metrics).sum()
                    label_moco = torch.tensor(0).type(torch.cuda.LongTensor)
                    # loss_infoNCE = loss_infoNCE + F.cross_entropy(metrics, label_moco)
                    loss_infoNCE_neg = F.cross_entropy(torch.index_select(metrics, 0, pure_index).unsqueeze(0), label_moco.unsqueeze(0))
                    # loss_sup_con = - (torch.log(torch.exp(torch.index_select(metrics, 0, pos_index)) / metric_exp_sum) * torch.index_select(sims, 0, pos_index)).sum()/weight_sum
                    
                    total_loss += loss_infoNCE_neg
                
                tl2.add(loss_meta)
                acc = count_acc(logits, label)

                tl1.add(total_loss.item())
                ta.add(acc)
                if self.mixed_precision:
                    with amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    total_loss.backward()

                self.optimizer.step() 
                
            self.lr_scheduler.step()
            self.logging(total_loss, loss_meta, acc)
            self.mini_test()
            self.save_model('epoch-last')
            if self.train_epoch%args.test100k_interval == 0:
                self.test_100k()

            print('ETA:{}/{}'.format(self.timer.measure(), self.timer.measure(self.train_epoch / args.max_epoch)))
    