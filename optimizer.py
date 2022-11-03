from torch.optim import SGD, AdamW, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, MultiStepLR, StepLR, ExponentialLR, \
    LambdaLR, SequentialLR

def get_optimizer(model, args):
    total_iter = args.epoch * args.iter_per_epoch
    warmup_iter = args.warmup_epoch * args.iter_per_epoch

    if args.optimizer == 'sgd':
        optimizer = SGD(model.parameter(), args.lr, args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.optimizer == 'adamw':
        optimizer = AdamW(model.parameter(), args.lr, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = RMSprop(model.parameter(), args.lr, eps=args.eps, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        NotImplementedError(f"{args.optimizer} is not supported yet")
    
    if args.scheduler == 'cosine':
        main_scheduler = CosineAnnealingLR(optimizer, total_iter-warmup_iter, args.min_lr)
    elif args.scheduler == 'cosinerestarts':
        main_scheduler = CosineAnnealingWarmRestarts(optimizer, total_iter // args.restart_epoch, 1, args.min_lr)
    elif args.scheduler == 'multistep':
        main_scheduler = MultiStepLR(optimizer, [epoch * args.iter_per_epoch for epoch in args.milestones])
    elif args.scheduler == 'step':
        main_scheduler = StepLR(optimizer, total_iter-warmup_iter, gamma=args.decay_rate)
    elif args.scheduler =='explr':
        main_scheduler = ExponentialLR(optimizer, gamma=args.decay_rate)
    else:
        NotImplementedError(f"{args.scheduler} is not supported yet")

    return optimizer, main_scheduler