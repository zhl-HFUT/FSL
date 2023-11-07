import numpy as np
import torch
from model.trainer.fsl_trainer import FSLTrainer
from model.utils import (
    pprint, set_gpu,
    get_command_line_parser,
    postprocess_args,
)
import json

# from ipdb import launch_ipdb_on_exception

if __name__ == '__main__':

    import random
    random.seed(1234) 
    np.random.seed(1234) 
    torch.manual_seed(1234) 
    torch.cuda.manual_seed(1234) 
    torch.cuda.manual_seed_all(1234)

    parser = get_command_line_parser()
    args = postprocess_args(parser.parse_args())
    # with launch_ipdb_on_exception():

    if args.config:
        config_dict = json.load(open(args.config, 'rb'))
        a = vars(args)
        a.update(config_dict)
    pprint(vars(args))

    set_gpu(args.gpu)

    # args.method = 'proto_net_only'
    # args.method = 'proto_FGKVM'
    args.method = 'MBT'
    # args.method = 'PMBT'
    if args.backbone_class == 'ConvNet':
        args.init_weights = './saves/mini_conv4_ver11_113120.pth'
        args.dim_model = 64
    elif args.backbone_class == 'Res12':
        args.init_weights = './saves/mini_r12_ver2_corrected_140403.pth'
        args.dim_model = 640

    trainer = FSLTrainer(args)

    if args.test is None:
        trainer.train()
    trainer.evaluate_test('max_acc.pth')
    trainer.evaluate_test('epoch-last.pth')
    trainer.final_record()
    print(args.save_path)
    trainer.wandb_run.finish()

    trainer.wandb_run.finish()
