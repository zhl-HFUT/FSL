import torch
import torch.nn as nn
import numpy as np

class BidirectionalLSTM(nn.Module):
    def __init__(self, layer_sizes, vector_dim):
        super(BidirectionalLSTM, self).__init__()

        self.hidden_size = layer_sizes[0]
        self.vector_dim = vector_dim
        self.num_layers = len(layer_sizes)

        self.lstm = nn.LSTM(input_size=self.vector_dim,
                            num_layers=self.num_layers,
                            hidden_size=self.hidden_size,
                            bidirectional=True)

    def forward(self, inputs, batch_size=1):
        c0 = torch.rand(self.lstm.num_layers*2, batch_size, self.lstm.hidden_size, requires_grad=False).cuda()
        h0 = torch.rand(self.lstm.num_layers*2, batch_size, self.lstm.hidden_size, requires_grad=False).cuda()
        output, (hn, cn) = self.lstm(inputs, (h0, c0))
        return output, hn, cn

class FewShotModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.task_feat=='output_max':
            self.register_buffer("queue", torch.randn(args.K, args.D*2))
        elif args.task_feat=='hn_mean':
            self.register_buffer("queue", torch.randn(args.K, args.D))
        self.lstm = BidirectionalLSTM(layer_sizes=[args.D], vector_dim = args.dim_model*args.spatial_dim*args.spatial_dim)
        self.K = args.K
        self.m = args.M
        self.T = args.T
        
        self.queue = nn.functional.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # classes of task in quene
        self.classes = np.ones((self.K, args.way), dtype=int)*1000

        if args.mem_init == 'random':
            memory_tensor = torch.randn(64, args.dim_model*args.spatial_dim*args.spatial_dim)
        elif args.mem_init == 'pre_train':
            memory_tensor = torch.load(args.mean_std)
            if args.mem_init_pooling == 'max':
                memory_tensor = nn.functional.max_pool2d(memory_tensor.reshape(64, -1, 5, 5), kernel_size=5)
            if args.mem_init_pooling == 'mean':
                memory_tensor = nn.functional.avg_pool2d(memory_tensor.reshape(64, -1, 5, 5), kernel_size=5)
        if args.mem_2d_norm:
            memory_tensor = memory_tensor.view(64, args.dim_model, args.spatial_dim, args.spatial_dim)
            memory_tensor = nn.functional.normalize(memory_tensor, dim=1)
            memory_tensor = memory_tensor.view(64, args.dim_model*args.spatial_dim*args.spatial_dim)
        if args.mem_grad:
            self.memory = nn.Parameter(memory_tensor.reshape(64, -1))
        else:
            self.register_buffer("memory", memory_tensor.reshape(64, -1))

        if args.backbone_class == 'ConvNet':
            from model.networks.convnet import ConvNet
            self.encoder = ConvNet(pooling=args.pooling)
            hdim = 64
        elif args.backbone_class == 'Res12':
            hdim = args.dim_model
            from model.networks.res12 import ResNet
            self.encoder = ResNet(drop_rate=args.drop_rate, out_dim=hdim, pooling=args.pooling)
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
            return  (torch.Tensor(np.arange(args.way*args.shot)).long().view(1, args.shot, args.way), 
                     torch.Tensor(np.arange(args.way*args.shot, args.way * (args.shot + args.query))).long().view(1, args.query, args.way))

    
    def forward(self, x, ids=None, simclr_images=None, key_cls=None, test=False):
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
            logits, logits_simclr, metrics, sims, pure_index, logits_blstm = self._forward(instance_embs, 
                support_idx, query_idx, key_cls=key_cls, ids=ids, simclr_embs=simclr_embs)
            return logits, logits_simclr, metrics, sims, pure_index, logits_blstm
        else:
            if test:
                origin_proto, proto, query = self._forward(instance_embs, support_idx, query_idx, ids, test=test)
                return origin_proto, proto, query
            logits, logits_blstm = self._forward(instance_embs, support_idx, query_idx, ids, test=test, key_cls=key_cls)
            return logits, logits_blstm

    def _forward(self, x, support_idx, query_idx):
        raise NotImplementedError('Suppose to be implemented by subclass')