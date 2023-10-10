# get a trainer like train_fsl.py for test
import numpy as np
import torch
from model.trainer.fsl_trainer import FSLTrainer
from model.utils import (
    set_gpu,
    get_command_line_parser,
    postprocess_args,
)
import numpy as np

import torch
import torch.nn.functional as F

from model.trainer.base import Trainer
from model.utils import (
    pprint, ensure_path,
    Averager, Timer, count_acc, one_hot,
    compute_confidence_interval,
    AccuracyClassAverager
)
from tqdm import tqdm
from shutil import copyfile
from json2html import *

def init_seed(seed):
    import random
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)

def get_parser():
    parser = get_command_line_parser()
    args = parser.parse_args([])
    args.max_epoch = 200
    args.model_class = 'FEATBaseTransformer3_2d'
    args.use_euclidean = True
    args.backbone_class = 'ConvNet'
    args.dataset = 'MiniImageNet'
    args.way = 5
    args.eval_way = 5
    args.shot = 1
    args.eval_shot = 1
    args.query = 15
    args.eval_query = 15
    args.balance = 0.0
    args.temperature = 0.1
    args.temperature2 = 0.1
    args.lr = 0.0001
    args.lr_mul = 10
    args.lr_scheduler = 'step'
    args.step_size = '20'
    args.gamma = 0.5
    args.gpu = '0'
    args.init_weights = './saves/mini_conv4_ver11_113120.pth'
    args.eval_interval = 1
    args.k = 30
    args.base_protos = 0
    args.feat_attn = 0
    args.pass_ids = 1
    args.base_wt = 0.1
    args.remove_instances = 1
    args.embed_pool = 'post_loss_avg'
    args.orig_imsize = 128
    args.fast_query = './embeds_cache/fastq_imgnet_wordnet_pathsim_random-preset-wts.pt'
    args.embeds_cache_2d = './embeds_cache/embeds_cache_cnn4_contrastive-init-ver1-1-corrected_2d.pt'
    args.wandb_mode = 'disabled'
    args.mixed_precision = 'O2'
    args.z_norm = 'before_tx'

    return args


def get_trainer(args):
    args = postprocess_args(args)
    set_gpu(args.gpu)
    trainer = FSLTrainer(args)

    return trainer

def custom_test(trainer, args, num_tasks=10000):
    import torch.nn as nn

    path = args.test
    params = torch.load(path)['params']
    del params['queue']
    print('deleted queue')
    trainer.model.load_state_dict(params, strict=False)
    trainer.model.eval()

    record = np.zeros(num_tasks) # loss and acc
    label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
    label = label.type(torch.LongTensor)
    if torch.cuda.is_available():
        label = label.cuda()

    with torch.no_grad():
        for i, batch in tqdm(enumerate(trainer.test_loader, 1)):
            if i == num_tasks+1:
                break
            data, gt_label, ids = batch[0].cuda(), batch[1].cuda(), batch[2]
            origin_proto, proto, query = trainer.model(data, ids, return_intermediate=True)

            # output, hn, cn = trainer.model.lstm(proto.view(5, 1, 1600).half())
            # feat_task_1 = hn.mean(dim = 0)
            # feat_task_1 = nn.functional.normalize(feat_task_1, dim=1) # (1, 256)

            # avgpool = nn.AdaptiveAvgPool1d(64)
            # avgpool_lstm_proto = avgpool(output)
            # avgpool_lstm_proto = avgpool_lstm_proto.view(1, 5, 64).unsqueeze(1).expand(1, 75, 5, 64).contiguous()
            # avgpool_lstm_proto = avgpool_lstm_proto.view(75, 5, 64)
            # avgpool_query = avgpool(query)
            # avgpool_query = avgpool_query.view(-1, 64).unsqueeze(1)

            # avgpool = nn.AdaptiveMaxPool1d(64)
            # avgpool_lstm_proto = avgpool(output)
            # avgpool_lstm_proto = avgpool_lstm_proto.view(1, 5, 64).unsqueeze(1).expand(1, 75, 5, 64).contiguous()
            # avgpool_lstm_proto = avgpool_lstm_proto.view(75, 5, 64)
            # # print(query.shape)
            # avgpool_query = avgpool(query.unsqueeze(1))
            # avgpool_query = avgpool_query.view(-1, 64).unsqueeze(1)

            # logits = - torch.mean((avgpool_lstm_proto - avgpool_query) ** 2, 2) / trainer.model.args.temperature

            # lstm_proto = output.view(1, 5, 512).unsqueeze(1).expand(1, 75, 5, 512).contiguous()
            # lstm_proto =lstm_proto.view(75, 5, 512)
            # feat_query = []
            # for emb_query in query:
            #     emb_lstm_query, _, _ = trainer.model.lstm(emb_query.view(1, 1, 1600).half())
            #     feat_query.append(emb_lstm_query)
            # lstm_query = torch.stack(feat_query).squeeze(1)
            # logits = - torch.mean((lstm_proto - lstm_query) ** 2, 2) / trainer.model.args.temperature

            # output, hn, cn = trainer.model.lstm(proto.view(5, 1, 1600).half())
            # feat_task_2 = hn.mean(dim = 0)
            # feat_task_2 = nn.functional.normalize(feat_task_2, dim=1) # (1, 256)
            # # print(origin_proto.shape, proto.shape, query.shape)

            query = query.view(-1, 1600).unsqueeze(1)
            proto = proto.unsqueeze(1).expand(1, 75, 5, 1600).contiguous()
            proto = proto.view(75, 5, 1600)
            origin_proto = origin_proto.view(1, 5, 1600).unsqueeze(1).expand(1, 75, 5, 1600).contiguous()
            origin_proto = origin_proto.view(75, 5, 1600)
            # print(origin_proto.shape, proto.shape, query.shape)

            logits1 = - torch.mean((origin_proto - query) ** 2, 2) / trainer.model.args.temperature
            logits2 = - torch.mean((proto - query) ** 2, 2) / trainer.model.args.temperature

            logits = - torch.mean((proto - query) ** 2, 2) / trainer.model.args.temperature
            
            acc = count_acc(logits, label)
            record[i-1] = acc
     
    # assert(i-1 == record.shape[0])
    va, vap = compute_confidence_interval(record)
    
    print('Test acc={:.4f} + {:.4f}\n'.format(va, vap))
    