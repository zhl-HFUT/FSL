import torch
import torch.nn as nn
import numpy as np
import sys
# from sal_utils import get_saliency_crops2, get_sal
from torch.cuda.amp import autocast


class BidirectionalLSTM(nn.Module):
    def __init__(self, layer_sizes, batch_size, vector_dim):
        super(BidirectionalLSTM, self).__init__()

        self.batch_size = batch_size
        self.hidden_size = layer_sizes[0]
        self.vector_dim = vector_dim
        self.num_layers = len(layer_sizes)

        self.lstm = nn.LSTM(input_size=self.vector_dim,
                            num_layers=self.num_layers,
                            hidden_size=self.hidden_size,
                            bidirectional=True)

    def forward(self, inputs):
        c0 = torch.rand(self.lstm.num_layers*2, self.batch_size, self.lstm.hidden_size, requires_grad=False).cuda().half()
        h0 = torch.rand(self.lstm.num_layers*2, self.batch_size, self.lstm.hidden_size, requires_grad=False).cuda().half()
        output, (hn, cn) = self.lstm(inputs, (h0, c0))
        return output, hn, cn


class FewShotModel(nn.Module):
    def __init__(self, args, resize=True, sal=False, max_pool='max_pool'):
        super().__init__()
        self.args = args

        self.lstm = BidirectionalLSTM(layer_sizes=[256], batch_size=1, vector_dim = 1600)
        self.K = 256
        self.m = 0.99
        self.T = 0.07
        self.register_buffer("queue", torch.randn(self.K, 256))
        self.queue = nn.functional.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # classes of task in quene
        self.classes = np.ones((self.K, 5), dtype=int)*1000

        if args.backbone_class == 'ConvNet':
            from model.networks.convnet import ConvNet
            self.encoder = ConvNet(resize=resize, sal=sal, max_pool=max_pool)
            hdim = 64
        else:
            raise ValueError('')
        self.sal_crop = args.sal_crop
        self.hdim = hdim

    @torch.no_grad()
    def _dequeue_and_enqueue(self, key, key_cls):
        ptr = int(self.queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr] = key
        self.classes[ptr] = key_cls
        # move pointer
        ptr = (ptr + 1) % self.K  
        self.queue_ptr[0] = ptr

    def split_instances(self, data):
        args = self.args
        if self.training:
            return  (torch.Tensor(np.arange(args.way*args.shot)).long().view(1, args.shot, args.way), 
                     torch.Tensor(np.arange(args.way*args.shot, args.way * (args.shot + args.query))).long().view(1, args.query, args.way))
        else:
            return  (torch.Tensor(np.arange(args.eval_way*args.eval_shot)).long().view(1, args.eval_shot, args.eval_way), 
                     torch.Tensor(np.arange(args.eval_way*args.eval_shot, args.eval_way * (args.eval_shot + args.eval_query))).long().view(1, args.eval_query, args.eval_way))

    
    def forward(self, x, ids=None,  get_feature=False, simclr_images=None, key_cls=None, return_intermediate=False):

        # with autocast():
            # input sal crop
        # if self.sal_crop is not None:
        #     if self.sal_crop == 'saliency_crop2':
        #         x = get_saliency_crops2(x,
        #                 use_minmax=False)
        #     elif self.sal_crop == 'saliency_crop4':
        #         sal = get_sal(x)
        #         sal = sal.repeat_interleave(3, dim=1)
        #         sal = sal*5.4-2.7
        #         # print('sal shape', [sal.shape, sal.min(), sal.max()])
        #         self.sal_embs= self.encoder(sal)
            
        # print('inside base', [x.shape, x.min(), x.max()])
        if get_feature:
            # get feature with the provided embeddings
            print('inside get_featur/e')
            return self.encoder(x)
        else:
            # feature extraction
            x = x.squeeze(0)
            instance_embs = self.encoder(x)
            num_inst = instance_embs.shape[0]
            # split support query set for few-shot data
            support_idx, query_idx = self.split_instances(x)
            # print('support and query idx')
            # print(support_idx)
            # print(query_idx)
            simclr_embs=None
            if simclr_images is not None:
                n_embs, n_views, n_ch, spatial, _ = simclr_images.shape
                simclr_images = simclr_images.reshape(-1, n_ch, spatial, spatial)
                simclr_embs = self.encoder(simclr_images)
                spatial_out = simclr_embs.shape[-1]
                # print('simclr embs ', [simclr_embs.shape, simclr_embs.min(), simclr_embs.max()])

                simclr_embs = simclr_embs.reshape(n_embs, n_views, self.hdim, spatial_out, spatial_out)
                # print('simclr embs ', [simclr_embs.shape, simclr_embs.min(), simclr_embs.max()])
                # print('instance embs ', [instance_embs.shape, instance_embs.min(), instance_embs.max()])

            if return_intermediate:
                origin_proto, proto, query = self._forward(instance_embs, 
                        support_idx, query_idx, key_cls=key_cls, ids=ids, simclr_embs=simclr_embs, return_intermediate=return_intermediate)
                return origin_proto, proto, query

            if self.training:
                if self.args.pass_ids:
                    logits, logits_reg, metrics, sims, pure_index = self._forward(instance_embs, 
                        support_idx, query_idx, key_cls=key_cls, ids=ids, simclr_embs=simclr_embs, return_intermediate=False)

                else:
                    logits, logits_reg = self._forward(instance_embs, 
                        support_idx, query_idx, simclr_embs=simclr_embs)
                return logits, logits_reg, metrics, sims, pure_index
            else:
                if self.args.pass_ids:
                    logits = self._forward(instance_embs, support_idx, query_idx, ids)
                else:
                    logits = self._forward(instance_embs, support_idx, query_idx)
                return logits

    def _forward(self, x, support_idx, query_idx):
        raise NotImplementedError('Suppose to be implemented by subclass')