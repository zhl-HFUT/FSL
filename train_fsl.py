import numpy as np
import torch
from model.trainer.fsl_trainer import FSLTrainer
from model.utils import (
    pprint, set_gpu,
    get_command_line_parser,
    postprocess_args,
)

if __name__ == '__main__':

    import random
    random.seed(1234) 
    np.random.seed(1234) 
    torch.manual_seed(1234) 
    torch.cuda.manual_seed(1234) 
    torch.cuda.manual_seed_all(1234)

    parser = get_command_line_parser()
    args = postprocess_args(parser.parse_args())

    args.fast = False

    set_gpu(args.gpu)

    print('-----------------------------')

    # 训练参数
    print('--backbone_class :', args.backbone_class)
    print('--metric :', args.metric)
    print('--spatial_dim :', args.spatial_dim)
    print('--pooling :', args.pooling)
    print('--shot :', args.shot)
    print('--lr :', args.lr)
    print('--step_size :', args.step_size)
    print('-----------------------------')
    
    # MemoryTransformers
    print('--mem_init :', args.mem_init)
    print('--mem_grad :', args.mem_grad)
    print('--mem_2d_norm :', args.mem_2d_norm)
    print('--mem_init_pooling :', args.mem_init_pooling)
    # simclr
    print('--use_simclr :', args.use_simclr)
    print('--balance :', args.balance)
    # BatchNorm
    print('--bn2d :', args.bn2d)
    print('-----------------------------')

    # infoNCE
    print('--use_infoNCE :', args.use_infoNCE)
    print('--task_feat :', args.task_feat)
    print('--T :', args.T)
    print('--K :', args.K)
    print('--D :', args.D)
    print('-----------------------------')

    # blstm分类
    print('--use_blstm_meta :', args.use_blstm_meta)
    print('--blstm_metric :', args.blstm_metric)
    print('--temperature3 :', args.temperature3)
    print('-----------------------------')

    print('\n\n\n-----------------------------')
    pprint(vars(args))
    
    trainer = FSLTrainer(args)

    trainer.train()
