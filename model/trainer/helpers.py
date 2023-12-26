import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model.dataloader.samplers import CategoriesSampler
from model.models.feat_basetransformer3_2d import FEATBaseTransformer3_2d

def get_dataloader(args):
    if args.dataset == 'MiniImageNet':
        from model.dataloader.mini_imagenet import MiniImageNet as Dataset

    num_episodes = args.episodes_per_epoch
    num_workers = args.num_workers
    trainset = Dataset('train', args, augment=False, 
        return_id=True, return_simclr=args.return_simclr)
    # ids to be passed to prevent base examples from the class of support instance not be considered 
    # by transformer.
    args.num_class = trainset.num_class
    train_sampler = CategoriesSampler(trainset.label,
                                      num_episodes,
                                      args.way,
                                      args.shot + args.query)

    train_loader = DataLoader(dataset=trainset,
                                  num_workers=num_workers,
                                  batch_sampler=train_sampler,
                                  pin_memory=True)

    valset = Dataset('val', args, return_id=True) 
    val_sampler = CategoriesSampler(valset.label,
                            args.num_eval_episodes,
                            args.way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset,
                            batch_sampler=val_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True)
    
    testset = Dataset('test', args, return_id=True)
    test_sampler = CategoriesSampler(testset.label,
                            10000, # args.num_eval_episodes,
                            args.way, args.shot + args.query)
    test_loader = DataLoader(dataset=testset,
                            batch_sampler=test_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True)    

    return train_loader, val_loader, test_loader

def get_update_loader(args, batch_size):
    from model.dataloader.mini_imagenet import MiniImageNet as Dataset
    trainset = Dataset('train', args, return_id=True)
    return DataLoader(trainset, shuffle=False, batch_size=batch_size)


def prepare_model(args):
    model = eval(args.model_class)(args)

    # load pre-trained model (no FC weights)
    state_dict=False
    if args.init_weights is not None:
        model_dict = model.state_dict()
        try:
            print('loading init_weights', args.init_weights)      
            pretrained_dict = torch.load(args.init_weights)['params']
        except:
            state_dict = True
            print('loading init_weights', args.init_weights)
            pretrained_dict = torch.load(args.init_weights)['state_dict']
        # print(pretrained_dict.keys())
        if args.backbone_class == 'ConvNet' and not state_dict:
            pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # print('pretrained dict keys after filtering', pretrained_dict.keys())
        
        model_dict.update(pretrained_dict)

        # print('model dict keys', model_dict.keys())
        print('loading state dict', model.load_state_dict(model_dict))

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(['device', device])
    model = model.to(device)

    return model

def prepare_optimizer(model, args):
    attn_para = [v for k,v in model.named_parameters() if 'attn' in k] 
    blstm_para = [v for k,v in model.named_parameters() if 'blstm' in k]

    # as in the literature, we use ADAM for ConvNet and SGD for other backbones
    if args.backbone_class == 'ConvNet':
        optimizer = optim.Adam(
            [{'params': model.encoder.parameters()},
             {'params': attn_para, 'lr': args.lr * args.lr_mul},
             {'params': blstm_para, 'lr': args.lr * args.lr_mul}],
            lr=args.lr,
            # weight_decay=args.weight_decay, do not use weight_decay here
        )                
    else:
        optimizer = optim.SGD(
            [{'params': model.encoder.parameters()},
             {'params': attn_para, 'lr': args.lr * args.lr_mul},
             {'params': blstm_para, 'lr': args.lr * args.lr_mul}],
            lr=args.lr,
            momentum=args.momentum,
            nesterov=True,
            weight_decay=args.weight_decay
        )        

    if args.lr_scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(
                            optimizer,
                            step_size=int(args.step_size),
                            gamma=args.gamma
                        )
    elif args.lr_scheduler == 'multistep':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
                            optimizer,
                            milestones=[int(_) for _ in args.step_size.split(',')],
                            gamma=args.gamma,
                        )
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            args.max_epoch,
                            eta_min=0   # a tuning parameter
                        )
    elif args.lr_scheduler == 'onecycle':
        print('here ')
        print([args.max_epoch,args.episodes_per_epoch ])
        lr_scheduler = optim.lr_scheduler.OneCycleLR(
                            optimizer,
                            max_lr=args.lr,
                            epochs=args.max_epoch,
                            steps_per_epoch=args.episodes_per_epoch   # a tuning parameter
                        )
    elif args.lr_scheduler == 'cyclic':
        print('here ')
        print([args.max_epoch,args.episodes_per_epoch ])
        lr_scheduler = optim.lr_scheduler.CyclicLR(
                            optimizer,
                            max_lr=args.lr,
                            base_lr=args.lr*1e-4,
                            step_size_up=args.episodes_per_epoch  # a tuning parameter
                        )
    else:
        raise ValueError('No Such Scheduler')

    return optimizer, lr_scheduler
