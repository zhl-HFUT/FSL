import os
import time
import pprint
import torch
import argparse
import numpy as np

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

# function to calculate class accuracies
# some way of getting class names in each iteration 

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
#     print('pred ', pred)
#     print('label ', label)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

def compute_confidence_interval(data):
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm

def postprocess_args(args):            
    args.num_classes = args.way
    save_path1 = '-'.join([args.dataset, args.model_class, args.backbone_class, '{:02d}w{:02d}s{:02}q'.format(args.way, args.shot, args.query)])
    save_path2 = '_'.join([str('_'.join(args.step_size.split(','))), str(args.gamma),
                           'lr{:.2g}mul{:.2g}'.format(args.lr, args.lr_mul),
                           str(args.lr_scheduler), 
                           'T1{}T2{}'.format(args.temperature, args.temperature2),
                           'b{}'.format(args.balance),
                           'bsz{:03d}'.format( max(args.way, args.num_classes)*(args.shot+args.query) )
                           ])    
    save_path1 += '-Pre'
    save_path1 += '-DIS'

    save_path2 += '_{}'.format(str(time.strftime('%Y%m%d_%H%M%S')))

    if not os.path.exists(os.path.join(args.save_dir, save_path1)):
        os.mkdir(os.path.join(args.save_dir, save_path1))
    args.save_path = os.path.join(args.save_dir, save_path1, save_path2)
    os.mkdir(args.save_path)
    return args

def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # 创新点相关
    parser.add_argument('--return_simclr', type=int, default=None) # number of views in simclr
    parser.add_argument('--use_infoNCE', action='store_true', default=False) # use infoNCE loss
    parser.add_argument('--n_heads', type=int, default=1) # self-attention heads
    parser.add_argument('--baseinstance_2d_norm', type=str, default=None)
    parser.add_argument('--z_norm', type=str, default=None, 
                        choices=['before_tx', 'before_euclidian', 'both', None])
    parser.add_argument('--balance', type=float, default=0)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--temperature2', type=float, default=0.1)

    # 测试，记录相关
    parser.add_argument('--wandb_mode', type=str, default='disabled')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--minitest_interval', type=int, default=1)
    parser.add_argument('--test100k_interval', type=int, default=20)
    
    # 模型
    parser.add_argument('--model_class', type=str, default='FEATBaseTransformer3_2d', 
                        choices=['MatchNet', 'ProtoNet', 'FEAT','FEATBaseTransformer3_2d'])
    parser.add_argument('--backbone_class', type=str, default='ConvNet',
                        choices=['ConvNet', 'Res12'])
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    
    # 数据集
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--episodes_per_epoch', type=int, default=100)
    parser.add_argument('--num_eval_episodes', type=int, default=600)
    parser.add_argument('--dataset', type=str, default='MiniImageNet',
                        choices=['MiniImageNet', 'TieredImageNet', 'TieredImageNet_og', 'CUB'])
    # 这里的resize分两步，先128再84，不知道为什么
    parser.add_argument('--im_size', type=int, default=128)
    parser.add_argument('--use_im_cache', action='store_true', default=False)
    
    # 学习率
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_mul', type=float, default=10)    
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['multistep', 'step', 'cosine', 'onecycle', 'cyclic'])
    parser.add_argument('--step_size', type=str, default='20')
    parser.add_argument('--gamma', type=float, default=0.5)    
    parser.add_argument('--momentum', type=float, default=0.9) #SGD momentum
    parser.add_argument('--drop_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0005)

    # 训练设置
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mixed_precision', type=str, default=None)

    return parser
