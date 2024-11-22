import math
import torch
def set_optimizer(opt, model, op_type='SGD'):
    if op_type=='SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr,
                                momentum=opt.momentum, weight_decay=opt.weight_decay)
    if op_type=='AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        
        
    return optimizer


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    eta_min = lr * (args.lr_decay_rate ** 3)
    lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.n_epochs)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr