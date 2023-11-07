import torch
import torch.nn as nn
import numpy as np
import pickle
import torch.nn.functional as F
from model import models
import os
import os.path as osp
from model.models import FewShotModel



def pairwise_distances_logits(query, proto, distance_type='euclid'):
    #  query n * dim
    #  prototype n_c * dim
    if distance_type == 'euclid':
        n = query.shape[0]
        m = proto.shape[0]
        distances = -((query.unsqueeze(1).expand(n, m, -1) -
                   proto.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)  
        
#         print(distances.shape)
        return distances
    elif distance_type == 'cosine':
        emb_dim = proto.shape[-1]
        proto = F.normalize(proto, dim=-1) # normalize for cosine distance
        query = query.view(-1, emb_dim) # (Nbatch,  Nq*Nw, d)
        
        # print([query.shape, proto.shape])
        logits = torch.matmul(query, proto.transpose(1,0))
#         print('logits.shape', logits.shape)
        return logits


def ids2classes(ids):
    # print('use object function')
    # raise NotImplementedError
    classes = np.array([id_[:-len('00000005')] for id_ in ids])
    return classes


def get_k_base(proto, all_proto, return_weights=False, k=10, 
               train=True, remove_instances=False, all_classes=None, current_classes=None,
               random=False):


    
    # remove_instances, all_classes, current_classes are specifically for baseinstances case 
    mask = np.zeros(all_classes.shape).astype(np.bool)

    # why this bit of code is required ?? while training in baseinstances case
    # in base_protos case this makes sense as you don't want itself to be in the combined_protos


    # all instances of the 5 classes are removed by remove_instances;
    if train and not remove_instances:
    # if train:
        start = 1
        end =0
    else:
        start = 0
        end = 1
        

    if random:
        mask = mask + 1
        a_ind = torch.randint(low=0, high=all_classes.shape[0], size=[proto.shape[0], k-1])
        return a_ind, mask


    if remove_instances:
        # remove all instances of the current class from all_proto
        filtered_ids = []
        for curr in current_classes:
            filtered_ids.append((np.argwhere(curr==all_classes)).squeeze())
        
        filtered_ids = np.concatenate(filtered_ids)
        mask = np.zeros(all_classes.shape).astype(np.bool)
 
        mask[filtered_ids] = 1

        all_proto = all_proto[~mask]
#         all_proto[mask] = 100
        
        start = 0
        end = 1 

    


    similarities = pairwise_distances_logits(proto, all_proto).squeeze()
    if similarities.dim() ==1:
        similarities = similarities.unsqueeze(0)
    similarities_sorted = torch.sort(similarities, descending=True, dim=1)

    a_ind = similarities_sorted.indices

    if return_weights:
        a = similarities_sorted.values
        a = F.softmax(a[:,start:], dim=1)
        return a_ind[:, start:k-end], a[:, :k-start-end], mask
    

    return a_ind[:, start:k-end], mask




class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        print('Creating transformer with d_k, d_v, d_model = ', [d_k, d_v, d_model])
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v):
        # print('here', [q.shape, k.shape, v.shape])
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        # print('here inside self attn q ', [type(q.detach().cpu().numpy()[0,0,0]), q.shape, q[0,0,0]])
        # print('here inside self attn k ', [type(k.detach().cpu().numpy()[0,0,0]), k.shape, k[0,0,0]])
        # print('here inside self attn v ', [type(v.detach().cpu().numpy()[0,0,0]), v.shape, v[0,0,0]])
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        # print('Inside multi head before self.attention ', [q.shape, k.shape, v.shape])

        output, attn, log_attn = self.attention(q, k, v)
        # print('here ', attn.shape)
        # print('log attn ', log_attn.shape)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output
    
class FEATBaseTransformer2(FewShotModel):
    def __init__(self, args):
        max_pool = args.max_pool
        resize = args.resize
        self.embeds_cache_root = './embeds_cache/'
        super().__init__(args, max_pool=max_pool, resize = resize)
        if args.backbone_class == 'ConvNet':
            hdim = 64
        elif args.backbone_class == 'Res12':
            hdim = args.dim_model
        else:
            raise ValueError('')
        

        tx_k_v = hdim
        print('Creating slf_attn with hdim, tx_k_v = ', [hdim, tx_k_v])
        self.slf_attn = MultiHeadAttention(args.n_heads, hdim, tx_k_v, tx_k_v, dropout=0.5)
        # proto_dict = None
        # self.fast_query = args.fast_query
        # if args.base_protos==0:
            
        #     # using base instances
        #     print('using base instances')
        #     print('backbone_class ', [args.backbone_class, args.backbone_class=='Res12_ptcv', args.dataset])
        #     if self.fast_query is None:
        #         if args.embeds_cache_1d is not None:
        #             print('loading 1d embeds_cache from args ', args.embeds_cache_1d)
        #             proto_dict = torch.load(args.embeds_cache_1d)
   
        #         self.all_proto = proto_dict['embeds'].cuda() 
        #         if self.args.mixed_precision is not None and self.args.mixed_precision!='O0':
        #             print('halving the embeds_cache 1d')
        #             self.all_proto = self.all_proto.half()
        #         self.proto_dict = proto_dict
        #         self.all_classes = self.ids2classes(np.array(proto_dict['ids']))
        # if self.fast_query is not None:
        #     print('Loading fast query_dict ', self.fast_query)
        #     print('path is ', self.fast_query)
        #     self.query_dict = torch.load(self.fast_query)
        #     # self.all_base_ids = np.array(list(self.query_dict.keys()))[:38400]
        self.after_attn = None
        self.top_k = None

        self.remove_instances = bool(self.args.remove_instances)
    
    def ids2classes(self, ids):
        if self.args.dataset=='MiniImagenet' or self.args.dataset=='TieredImageNet_og':
            classes = np.array([id_[:-len('00000005')] for id_ in ids])
            return classes
        elif self.args.dataset=='CUB':
            classes = np.array(['_'.join(id_.split('_')[:-2]) for id_ in ids])
        else:
            raise NotImplementedError
        return classes

    def _forward(self, instance_embs, support_idx, query_idx, ids=None):
        raise NotImplementedError
