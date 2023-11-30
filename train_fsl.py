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

    set_gpu(args.gpu)

    pprint(vars(args))

    trainer = FSLTrainer(args)

    trainer.train()
