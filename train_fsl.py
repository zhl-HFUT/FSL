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

    pprint(vars(args))

    set_gpu(args.gpu)

    if args.backbone_class == 'ConvNet':
        args.init_weights = './saves/mini_conv4_ver11_113120.pth'
        args.mean_std = './.cache/mean_std_conv4.pth'
        args.dim_model = 64
        args.dim_hn = 256
    elif args.backbone_class == 'Res12':
        args.init_weights = './saves/mini_r12_ver2_corrected_140403.pth'
        args.mean_std = './.cache/mean_std_res12.pth'
        args.dim_model = 640
        args.dim_hn = 256

    trainer = FSLTrainer(args)

    trainer.train()
    if args.test100k_interval != 1:
        trainer.evaluate_test('max_acc.pth')
    trainer.final_record()
    print(args.save_path)
