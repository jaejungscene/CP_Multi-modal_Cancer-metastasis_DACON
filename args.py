import argparse
from datetime import datetime
result_folder_name = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def get_args_parser():
    parser = argparse.ArgumentParser(description='training CIFAR-10, CIFAR-100 for self-directed research')
    parser.add_argument('--model', default='resnet', type=str, help='networktype: resnet')
    parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', default=0.1, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')                    
    parser.add_argument('--cuda', type=str, default='0', help='select used GPU')
    parser.add_argument('--wandb', type=int, default=1, help='choose activating wandb')
    parser.add_argument('--optim', type=str, default='sgd', help='select optimizer')
    parser.add_argument('--seed', type=str, default=41, help='set seed')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--expname', default=result_folder_name, type=str, help='name of experiment')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='W', help='number of data loading workers (default: 4)')


    optimizer = parser.add_argument_group('optimizer')
    optimizer.add_argument('--lr', type=float, default=1e-3, help='learning rate(lr)')
    optimizer.add_argument('--epochs', type=int, default=100, help='epoch')
    optimizer.add_argument('--optimizer', type=str, default='adamw', help='optimizer name')
    optimizer.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
    optimizer.add_argument('--weight-decay', type=float, default=1e-3, help='optimizer weight decay')
    optimizer.add_argument('--nesterov', action='store_true', default=False, help='use nesterov momentum')
    optimizer.add_argument('--betas', type=float, nargs=2, default=[0.9, 0.999], help='adam optimizer beta parameter')
    optimizer.add_argument('--eps', type=float, default=1e-6, help='optimizer eps')
    optimizer.add_argument('--min-lr', type=float, default=1e-6, help='lowest lr used for cosine scheduler')
    optimizer.add_argument('--decay-rate', type=float, default=0.1, help='lr decay rate')
    optimizer.add_argument('--restart-epoch', type=int, default=20, help='warmup restart epoch period')
    optimizer.add_argument('--warmup-epoch', type=int, default=5, help='warmup epoch')

    parser.add_argument('--distil', type=int, default=0, help='choose whether to do knowledge distillation')
    parser.add_argument('--distil_type', type=str, default='hard', help='choose what type of knowledge distillation')
    # parser.add_argument('--alpha', default=300, type=float,
    #                     help='number of new channel increases per depth (default: 300)')
    # parser.add_argument('--beta', default=0, type=float,
    #                     help='hyperparameter beta')
    # parser.add_argument('--cutmix_prob', default=0, type=float,
    #                     help='cutmix probability')

    return parser