from model.models import FewShotModel
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import random

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

class FEATBaseTransformer3_2d(FewShotModel):
    def __init__(self, args):
        super().__init__(args)
        self.slf_attn = MultiHeadAttention(args.n_heads, args.dim_model, args.dim_model, args.dim_model, dropout=0.5)
        
        import json
        with open('wordnet_sim_labels.json', 'r') as file:
            data = json.load(file)
        self.wordnet_sim_labels = data
            
        self.embed_pool = torch.nn.Identity()
 
        self.label_aux = None

        self.choose = None
        self.reshape_dim = None

        self.baseinstance_2d_norm = None
        if args.baseinstance_2d_norm:
            self.baseinstance_2d_norm = nn.BatchNorm2d(self.hdim)

    def get_simclr_logits(self, simclr_features, temperature_simclr, fc_simclr=None, max_pool=False):

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

        spatial_dim = 5
        n_class = 5
        n_simcls = 5
        n_batch = 1
        emb_dim = self.args.dim_model

        support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (emb_dim, spatial_dim, spatial_dim))) # 1,1,5,64,5,5
        query   = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(*(query_idx.shape + (emb_dim, spatial_dim, spatial_dim))) # 1,15,5,64,5,5
        proto = support.mean(dim=1) # 1,5,64,5,5
        
        if self.training:
            self.wordnet_sim_labels['n0461250400'][4] = random.choice([21, 49, 52, 40])
            top_indices = np.stack([self.wordnet_sim_labels[id_[:11]] for id_ in ids[:5]], axis=0)
            base_protos = self.memory[torch.Tensor(top_indices).long()].reshape(n_class, n_simcls, emb_dim, spatial_dim, spatial_dim)
        else:
            top_indices = np.stack([self.wordnet_sim_labels[id_[:11]] for id_ in ids[:5]], axis=0)
            base_protos = self.memory[torch.Tensor(top_indices).long()].reshape(n_class, n_simcls, emb_dim, spatial_dim, spatial_dim)

        if self.baseinstance_2d_norm:
            base_protos = base_protos.reshape(n_class*n_simcls, emb_dim, spatial_dim, spatial_dim)
            base_protos = self.baseinstance_2d_norm(base_protos)
            base_protos = base_protos.reshape(n_class, n_simcls, emb_dim, spatial_dim, spatial_dim)

        if self.args.z_norm=='before_tx':
            base_protos = base_protos.reshape(n_class*n_simcls, emb_dim*spatial_dim*spatial_dim)
            proto = proto.reshape(n_class, emb_dim*spatial_dim*spatial_dim)
            base_protos, proto = apply_z_norm(base_protos), apply_z_norm(proto)                                                                                                
            base_protos = base_protos.reshape(n_class, n_simcls, emb_dim, spatial_dim, spatial_dim)
            proto = proto.reshape(n_batch, n_class, emb_dim, spatial_dim, spatial_dim)

        origin_proto = proto.view(5, 1, emb_dim*spatial_dim*spatial_dim).contiguous()

        proto = proto.view(5, emb_dim, spatial_dim*spatial_dim).permute(0, 2, 1).contiguous()
        combined_bases = base_protos.permute(0, 1, 3, 4, 2).reshape(n_class, n_simcls*spatial_dim*spatial_dim, emb_dim).contiguous() # 5,125,64

        # 在此做attention增强
        proto = self.slf_attn(proto, combined_bases, combined_bases) # 5,25,64
        proto = proto.permute(0, 2, 1).contiguous() # 5,64,25

        # 计算logits-meta
        proto = proto.view(1, -1, emb_dim*spatial_dim*spatial_dim).contiguous() # 1,5,1600
        query = query.view(-1, 1, emb_dim*spatial_dim*spatial_dim).contiguous() # 75,1,1600
        logits = - torch.mean((proto - query) ** 2, 2) / self.args.temperature

        # task feature部分
        if self.training:
            # attention之前的任务特征
            output, hn, cn = self.lstm(origin_proto)
            feat_task_1 = hn.mean(dim = 0)
            feat_task_1 = nn.functional.normalize(feat_task_1, dim=1) # (1, 256)

            # attention之后的任务特征
            output, hn, cn = self.lstm(proto.permute(1, 0, 2))
            feat_task_2 = hn.mean(dim = 0)
            feat_task_2 = nn.functional.normalize(feat_task_2, dim=1) # (1, 256)

            # 计算metrics
            metric_pos = torch.dot(feat_task_2.squeeze(0), feat_task_1.squeeze(0)).unsqueeze(-1)
            metric_memory = torch.mm(feat_task_2, self.queue.clone().detach().t())
            metrics = torch.cat((metric_pos, metric_memory.squeeze(0)), dim=0)
            metrics /= self.T

            # 计算task overlap sims，得到纯负样本index
            sims = [1]
            pure_index = [0]
            for i in range(self.K):
                sims.append(len(np.intersect1d(self.classes[i,:], key_cls.cpu()))/5.)
                if not bool(len(np.intersect1d(self.classes[i,:], key_cls.cpu()))):
                    pure_index.append(i+1)
            
            # attention之后的任务特征入队
            self._dequeue_and_enqueue(feat_task_2, key_cls.cpu())

            if self.args.use_memNorm:
                loss_mem = 0
                for mem in self.memory:
                    A = mem.reshape(25, 64)
                    # print(A)
                    B = torch.mm(A, A.t()) - torch.eye(A.shape[0]).cuda()
                    # eigenvalues = torch.linalg.eigvals(torch.matmul(B.t(), B))
                    eigenvalues = torch.linalg.eigvals(B)
                    max_eigenvalue = torch.max(torch.real(eigenvalues))
                    # loss_mem += torch.sqrt(max_eigenvalue)
                    loss_mem += max_eigenvalue
                print(loss_mem)
                # print(logits)

        # simclr logits部分
        if self.training:
            if  simclr_embs is not None:
                fc_simclr = None
                logits_simclr = self.get_simclr_logits(simclr_embs,
                    temperature_simclr=self.args.temperature2,
                    fc_simclr=fc_simclr) 
                return logits, logits_simclr, metrics, sims, pure_index
            # 训练时在这里return
            if self.args.balance==0:
                return logits, None, metrics, sims, pure_index, loss_mem
        # 测试时在这里return
        else:
            return logits