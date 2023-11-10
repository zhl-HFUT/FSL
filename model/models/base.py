import torch
import torch.nn as nn
import numpy as np
import sys

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
    def __init__(self, args, resize=True, max_pool='max_pool'):
        super().__init__()
        self.args = args

        self.lstm = BidirectionalLSTM(layer_sizes=[args.dim_hn], batch_size=1, vector_dim = args.dim_model*25)
        self.K = 256
        self.m = 0.99
        self.T = 0.07
        self.register_buffer("queue", torch.randn(self.K, args.dim_hn))
        self.queue = nn.functional.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # classes of task in quene
        self.classes = np.ones((self.K, 5), dtype=int)*1000

        self.memory = nn.Parameter(torch.load('/BaseTransformers/mean_std.pth'))

        if args.backbone_class == 'ConvNet':
            from model.networks.convnet import ConvNet
            self.encoder = ConvNet()
            hdim = 64
        elif args.backbone_class == 'Res12':
            hdim = args.dim_model
            from model.networks.res12 import ResNet
            self.encoder = ResNet(avg_pool=False, 
                drop_rate=args.drop_rate, out_dim=hdim)
        else:
            raise ValueError('')
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

    def split_instances(self):
        args = self.args
        if self.training:
            return  (torch.Tensor(np.arange(args.way*args.shot)).long().view(1, args.shot, args.way), 
                     torch.Tensor(np.arange(args.way*args.shot, args.way * (args.shot + args.query))).long().view(1, args.query, args.way))
        else:
            return  (torch.Tensor(np.arange(args.eval_way*args.eval_shot)).long().view(1, args.eval_shot, args.eval_way), 
                     torch.Tensor(np.arange(args.eval_way*args.eval_shot, args.eval_way * (args.eval_shot + args.eval_query))).long().view(1, args.eval_query, args.eval_way))

    
    def forward(self, x, ids=None, simclr_images=None, key_cls=None):
        # feature extraction
        x = x.squeeze(0)
        instance_embs = self.encoder(x)
        # split support query set for few-shot data
        support_idx, query_idx = self.split_instances()
        simclr_embs = None
        if simclr_images is not None:
            n_embs, n_views, n_ch, spatial, _ = simclr_images.shape
            simclr_images = simclr_images.reshape(-1, n_ch, spatial, spatial)
            simclr_embs = self.encoder(simclr_images)
            spatial_out = simclr_embs.shape[-1]
            simclr_embs = simclr_embs.reshape(n_embs, n_views, self.hdim, spatial_out, spatial_out)

        if self.training:
            logits, logits_simclr, metrics, sims, pure_index = self._forward(instance_embs, 
                support_idx, query_idx, key_cls=key_cls, ids=ids, simclr_embs=simclr_embs)
            return logits, logits_simclr, metrics, sims, pure_index
        else:
            logits = self._forward(instance_embs, support_idx, query_idx, ids)
            return logits

    def _forward(self, x, support_idx, query_idx):
        raise NotImplementedError('Suppose to be implemented by subclass')