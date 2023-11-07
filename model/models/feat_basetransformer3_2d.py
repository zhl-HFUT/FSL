from .feat_basetransformer3 import FEATBaseTransformer3
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import random

def apply_z_norm(features):                                                                                                                                                    
    d = features.shape[-1]                                                                                                                                                     
    features_mean = features.mean(dim=-1)                                                                                                                                      
    features_std = features.std(dim=-1)                                                                                                                                        
    # print([features_mean.mean(), features_std.mean()])                                                                                                                       
    features_znorm = (features-features_mean.unsqueeze(1).repeat(1, d))/(features_std.unsqueeze(1).repeat(1, d))                                                               
    return features_znorm  

def get_choose(n_key, n_classes=5):

    choose = torch.diag(torch.ones(n_key))
    
    # now remove all the 5th elements on either side of the diagonal 
    # except for the top right quadrant diagonal and bottom left diagonal
    # because these are the distances (ai-bi) which are the only positives in each row
    
    positive_choose = torch.zeros(n_key, n_key)

    indices = torch.arange(0,n_key,n_classes)
#     print('indices ', indices)
    n_half = int(indices.shape[0]/2)
#     print('n_half', [n_half, indices[n_half]])
    indices_selected = torch.cat([indices[0:n_half], indices[n_half+1:]])
#     print(indices_selected, 'indices_selected')
    positive_index = indices[n_half]
    positive_choose_0 = torch.zeros(n_key)
    positive_choose_0[positive_index] = 1
    choose[0, indices_selected] = 1
    choose_0 = choose[0,:]
    choose_list = []
    positive_choose_list = []
    label_list = []
    for i in range(n_key):
#         print(['i', 'positive', i, torch.argmax(positive_choose_0)])
#         print('choose_0 ', choose_0)
        choose_list.append(choose_0.unsqueeze(0))
        label_list.append(torch.argmax(positive_choose_0).item())
        positive_choose_list.append(positive_choose_0.unsqueeze(0))

        choose_0 = torch.roll(choose_0, 1, dims=0)
        positive_choose_0 = torch.roll(positive_choose_0, 1, dims=0)

#     print(label_list)
    choose = torch.cat(choose_list)
    positive_choose = torch.cat(positive_choose_list)
    label = torch.Tensor(label_list)
    return choose, positive_choose

class FEATBaseTransformer3_2d(FEATBaseTransformer3):
    def __init__(self, args):
        args.max_pool = False
        args.resize = False
        super().__init__(args)
        # these 2d embeddings of base instances are used for combination
        
        import json
        with open('wordnet_sim_labels.json', 'r') as file:
            data = json.load(file)
        self.wordnet_sim_labels = data
            
        self.embed_pool = torch.nn.Identity()
 
        self.spatial_dim = None # will be initialized in forward
        self.label_aux = None

        self.choose = None
        self.reshape_dim = None

        self.baseinstance_2d_norm = None
        if args.baseinstance_2d_norm:
            self.baseinstance_2d_norm = nn.BatchNorm2d(self.hdim)

    def get_simclr_logits(self, simclr_features, temperature_simclr, fc_simclr=None, max_pool=False, version=None):

        # print('simclr_features ', simclr_features.shape)
        
        n_batch, n_views, n_c, spatial, _ = simclr_features.shape

        if fc_simclr is not None:
            simclr_features = fc_simclr(simclr_features)

        max_pool=True
        if max_pool:
            simclr_features = simclr_features.reshape(n_batch*n_views, n_c, spatial, spatial)
            simclr_features = F.max_pool2d(simclr_features, kernel_size=5)
            simclr_features = simclr_features.reshape(n_batch, n_views, n_c, 1, 1)

        simclr_features = simclr_features.reshape(n_batch, n_views, -1)

        # now calculate logits using euclidian loss or cosine loss;

        a = torch.cat([simclr_features[:, 0, :], simclr_features[:, 1, :]], dim=0)
        b = torch.cat([simclr_features[:, 0, :], simclr_features[:, 1, :]], dim=0)

        n_key, emb_dim = a.shape
        n_query = b.shape[0]
        a = a.unsqueeze(0).expand(n_query, n_key, emb_dim)
        b = b.unsqueeze(1)
        logits_simclr = - torch.mean((a - b) ** 2, 2) / temperature_simclr

        # remove diagonal elements

        n_classes = 5
        if self.label_aux is None:
            choose, positive_choose = get_choose(n_key, n_classes)
            self.reshape_dim = int(choose.sum(1)[0].item())
            choose = ~(choose.bool())
            label_aux = positive_choose[choose].reshape(n_query, n_key-self.reshape_dim).argmax(1)
            self.label_aux = label_aux.cuda()
            self.choose = choose

        logits_simclr = logits_simclr[self.choose].reshape(n_query, n_key-self.reshape_dim)

        return logits_simclr

    def _forward(self, instance_embs, support_idx, query_idx, ids=None, simclr_embs=None, key_cls=None):

        spatial_dim = instance_embs.shape[-1]
        self.spatial_dim = spatial_dim
        emb_dim = instance_embs.size(-3)

        support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (emb_dim, spatial_dim, spatial_dim,)))
        query   = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(  *(query_idx.shape   + (emb_dim, spatial_dim, spatial_dim,)))

        proto = support.mean(dim=1) 
        n_class = proto.shape[1]
        n_batch = proto.shape[0]

        if self.training:
            self.wordnet_sim_labels['n0461250400'][4] = random.choice([21, 49, 52, 40])
            top_indices = np.stack([self.wordnet_sim_labels[id_[:11]] for id_ in ids[:5]], axis=0)
            base_protos = self.memory[torch.Tensor(top_indices).long()].reshape(5, 5, 640)
        else:
            top_indices = np.stack([self.wordnet_sim_labels[id_[:11]] for id_ in ids[:5]], axis=0)
            base_protos = self.memory[torch.Tensor(top_indices).long()].reshape(5, 5, 640)

        # if self.baseinstance_2d_norm:
        #     base_protos = base_protos.reshape(n_class*k, emb_dim, spatial_dim, spatial_dim)
        #     base_protos = self.baseinstance_2d_norm(base_protos)
        #     base_protos = base_protos.reshape(n_class, k, emb_dim, spatial_dim, spatial_dim)

        if self.args.z_norm=='before_tx' or self.args.z_norm=='both':                                                                                                          
                                                                                                                                                                               
            # b1, b2, b3, b4, _ = base_protos.shape                                                                                                                              
            # p1, p2, p3, p4, _ = proto.shape                                                                                                                                    
            # base_protos = base_protos.reshape(b1*b2, b3*b4*b4)                                                                                                                 
            # proto = proto.reshape(p1*p2, p3*p4*p4)                                                                                                                             

            base_protos = base_protos.reshape(25, 640)                                                                                                              
            proto = proto.reshape(5, 640)
            base_protos, proto = apply_z_norm(base_protos), apply_z_norm(proto)                                                                                                
            # base_protos = base_protos.reshape(b1,b2,b3,b4,b4)                                                                                                                  
            # proto = proto.reshape(p1,p2,p3,p4,p4)   
            origin_proto = proto.view(5, 1, 640)                                                                                                                           

        # proto = proto.reshape(proto.shape[1], emb_dim, -1).permute(0, 2, 1).contiguous()
        proto = proto.reshape(5, 640, 1).permute(0, 2, 1).contiguous()

        # base_protos = base_protos.permute(0, 2, 1, 3, 4).contiguous()
        combined_protos = base_protos.reshape(5, 640, 5).permute(0, 2, 1).contiguous()

        if self.args.method == 'proto_net_only':
            proto = proto.permute(0, 2, 1).contiguous()
            proto = proto.view(-1, emb_dim, spatial_dim, spatial_dim)
            query = query.view(-1, emb_dim, spatial_dim, spatial_dim)
            proto = proto.reshape(proto.shape[0], -1).unsqueeze(0)
            query = query.reshape(query.shape[0], -1)
            emb_dim = emb_dim*(spatial_dim**2)
            num_batch = proto.shape[0]
            num_proto = proto.shape[1]
            num_query = np.prod(query_idx.shape[-2:])
            # print(proto.shape, query.shape)

            query = query.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
            proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
            proto = proto.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)
            # print(proto.shape, query.shape)

            logits = - torch.mean((proto - query) ** 2, 2) / self.args.temperature
            return logits

        proto = self.slf_attn(proto, combined_protos, combined_protos)
        proto = proto.permute(0, 2, 1).contiguous()

        proto = proto.view(-1, emb_dim, spatial_dim, spatial_dim)
        query = query.view(-1, emb_dim, spatial_dim, spatial_dim)

        proto = proto.reshape(proto.shape[0], -1).unsqueeze(0)
        query = query.reshape(query.shape[0], -1)
        emb_dim = emb_dim*(spatial_dim**2)
       
            
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = np.prod(query_idx.shape[-2:])

        self.after_attn = proto #（1，5，1600）

        if self.training:
            # 此处过blstm
            output, hn, cn = self.lstm(origin_proto)
            feat_task_1 = hn.mean(dim = 0)
            feat_task_1 = nn.functional.normalize(feat_task_1, dim=1) # (1, 256)

            output, hn, cn = self.lstm(proto.view(5, 1, 640))
            feat_task_2 = hn.mean(dim = 0)
            feat_task_2 = nn.functional.normalize(feat_task_2, dim=1) # (1, 256)

            # 计算metrics, sims等
            metric_pos = torch.dot(feat_task_2.squeeze(0), feat_task_1.squeeze(0)).unsqueeze(-1)
            metric_memory = torch.mm(feat_task_2, self.queue.clone().detach().t())
            metrics = torch.cat((metric_pos, metric_memory.squeeze(0)), dim=0)
            metrics /= self.T

            sims = [1]
            pure_index = [0]
            for i in range(self.K):
                sims.append(len(np.intersect1d(self.classes[i,:], key_cls.cpu()))/5.)
                if not bool(len(np.intersect1d(self.classes[i,:], key_cls.cpu()))):
                    pure_index.append(i+1)
            
            # 入队出队
            self._dequeue_and_enqueue(feat_task_2, key_cls.cpu())

        query = query.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
        proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
        proto = proto.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)

        logits = - torch.mean((proto - query) ** 2, 2) / self.args.temperature

        # for regularization
        if self.training:
            # return logits, None
            # TODO this can be further adapted for basetransformer version
                # implementing simclr loss on the encoder embeddings
            if  simclr_embs is not None:

                if self.args.simclr_loss_type=='ver2.2':
                    fc_simclr = None

                    logits_simclr = self.get_simclr_logits(simclr_embs,
                        temperature_simclr=self.args.temperature2,
                        fc_simclr=fc_simclr,
                        version=self.args.simclr_loss_type) 
                return logits, logits_simclr, metrics, sims, pure_index
            # 训练时在这里return
            if self.args.balance==0:

                return logits, None, metrics, sims, pure_index
            aux_task = torch.cat([support.view(1, self.args.shot, self.args.way, emb_dim), 
                                  query.view(1, self.args.query, self.args.way, emb_dim)], 1) # T x (K+Kq) x N x d
            num_query = np.prod(aux_task.shape[1:3])
            aux_task = aux_task.permute([0, 2, 1, 3])
            aux_task = aux_task.contiguous().view(-1, self.args.shot + self.args.query, emb_dim)
            # apply the transformation over the Aug Task
            if self.feat_attn==1:
                aux_emb = self.self_attn2(aux_task, aux_task, aux_task) # T x N x (K+Kq) x d
            else:
                aux_emb = self.slf_attn(aux_task, combined_protos, combined_protos)
            # compute class mean
            aux_emb = aux_emb.view(num_batch, self.args.way, self.args.shot + self.args.query, emb_dim)
            aux_center = torch.mean(aux_emb, 2) # T x N x d
            
            aux_task = aux_task.permute([1,0,2]).contiguous().view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
            aux_center = aux_center.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
            aux_center = aux_center.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)

            logits_reg = - torch.sum((aux_center - aux_task) ** 2, 2) / self.args.temperature2        
            # 训练时在这里return
            return logits, logits_reg            
        # 测试时在这里return
        else:
            return logits