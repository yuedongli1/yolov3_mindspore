import math
import numpy as np


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def get_group_param_yolov3(model):
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for p in model.trainable_params():
        if "bias" in p.name:
            pg2.append(p)  # biases
        if "bn.gamma" in p.name in p.name:
            pg0.append(p) # no decay
        elif "weight" in p.name:
            pg1.append(p) # apply decay

    return pg0, pg1, pg2


def get_lr_yolov3(opt, hyp, per_epoch_size):
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    init_lr, warmup_bias_lr, warmup_epoch, lrf = \
        hyp['lr0'], hyp['warmup_bias_lr'], hyp['warmup_epochs'], hyp['lrf']
    total_epoch, linear_lr = opt.epochs, opt.linear_lr
    if opt.optimizer == "sgd":
        with_momentum = True
    elif opt.optimizer == "adam":
        with_momentum = False
    else:
        raise NotImplementedError

    if linear_lr:
        lf = lambda x: (1 - x / (total_epoch - 1)) * (1.0 - lrf) + lrf  # linear
    else:
        lf = one_cycle(1, lrf, total_epoch)  # cosine 1->hyp['lrf']

    lr_pg0, lr_pg1, lr_pg2 = [], [], []
    momentum_pg = []
    warmup_steps = max(round(warmup_epoch * per_epoch_size), 1000)
    xi = [0, warmup_steps]
    for i in range(total_epoch * per_epoch_size):
        cur_epoch = i // per_epoch_size
        _lr = init_lr * lf(cur_epoch)
        if i < warmup_steps:
            lr_pg0.append(np.interp(i, xi, [0.0, _lr]))
            lr_pg1.append(np.interp(i, xi, [0.0, _lr]))
            lr_pg2.append(np.interp(i, xi, [warmup_bias_lr, _lr]))
            if with_momentum:
                momentum_pg.append(np.interp(i, xi, [hyp['warmup_momentum'], hyp['momentum']]))
        else:
            lr_pg0.append(_lr)
            lr_pg1.append(_lr)
            lr_pg2.append(_lr)

    return lr_pg0, lr_pg1, lr_pg2, momentum_pg, warmup_steps