import yaml
import math
import os
import random
import time
import numpy as np
from pathlib import Path

import mindspore as ms
from mindspore import nn, ops, context, Tensor
from mindspore.amp import DynamicLossScaler, all_finite
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size

from network.yolo import Model
from network.common import ModelEMA
from network.loss import ComputeLoss
from config.args import get_args_train
from utils.optimizer import get_group_param_yolov3, get_lr_yolov3
from utils.dataset import create_dataloader
from utils.general import increment_path, colorstr, labels_to_class_weights, check_file, check_img_size, all_finite_cpu


def train(hyp, opt):
    ms.set_seed(2)
    save_dir, epochs, batch_size, total_batch_size, weights, rank, freeze = \
        opt.save_dir, opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.rank, opt.freeze

    with open(opt.data, 'rb') as f:
        data_dict = yaml.safe_load(f)  # data dict
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Directories
    wdir = os.path.join(save_dir, "weights")
    os.makedirs(wdir, exist_ok=True)
    # Save run settings
    with open(os.path.join(save_dir, "hyp.yaml"), 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(os.path.join(save_dir, "opt.yaml"), 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Model
    sync_bn = opt.sync_bn and context.get_context("device_target") == "Ascend" and rank_size > 1
    model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors'), sync_bn=sync_bn)  # create
    if rank % 8 == 0:
        ema_model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors'), sync_bn=sync_bn)
        ema = ModelEMA(ema_model)
    else:
        ema = None

    pretrained = weights.endswith('.ckpt')
    if pretrained:
        param_dict = ms.load_checkpoint(weights)
        ms.load_param_into_net(model, param_dict)
        print(f"Pretrain model load from {weights}")
        if ema and opt.ema_weight.endswith('.ckpt'):
            param_dict_ema = ms.load_checkpoint(opt.ema_weight)
            ms.load_param_into_net(ema.ema_model, param_dict_ema)
            ema.updates = param_dict_ema["updates"]
            print(f"Ema model load from {opt.ema_weight}")

    # Freeze
    freeze = [f'model.{x}.' for x in
              (freeze if len(freeze) > 1 else range(freeze[0]))]  # parameter names to freeze (full or partial)
    for n, p in model.parameters_and_names():
        if any(x in n for x in freeze):
            print('freezing %s' % n)
            p.requires_grad = False

    # Image sizes
    gs = max(int(model.stride.asnumpy().max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples
    train_path = "D:/yolov3_fromv7/datasets/coco128"
    train_path = "D:/yolov3_fromv7/datasets/coco/train2017.txt"
    train_path = data_dict['train']
    dataloader, dataset, per_epoch_size = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                                            hyp=hyp, augment=True, cache=opt.cache_images,
                                                            rect=opt.rect, rank_size=opt.rank_size, rank=opt.rank,
                                                            num_parallel_workers=1,
                                                            image_weights=opt.image_weights, quad=opt.quad,
                                                            prefix=colorstr('train: '))
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    print(f"Scaled weight_decay = {hyp['weight_decay']}")
    if "yolov3" in opt.cfg:
        pg0, pg1, pg2 = get_group_param_yolov3(model)
        lr_pg0, lr_pg1, lr_pg2, momentum_pg, warmup_steps = get_lr_yolov3(opt, hyp, per_epoch_size)
        assert len(momentum_pg) == warmup_steps
        momentum_pg = Tensor(momentum_pg, ms.float32)
    else:
        raise NotImplementedError
    group_params = [{'params': pg0, 'lr': lr_pg0},
                    {'params': pg1, 'lr': lr_pg1, 'weight_decay': hyp['weight_decay']},
                    {'params': pg2, 'lr': lr_pg2}]
    if opt.optimizer == "sgd":
        optimizer = nn.SGD(group_params, learning_rate=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    elif opt.optimizer == "adam":
        optimizer = nn.Adam(group_params, learning_rate=hyp['lr0'], beta1=hyp['momentum'], beta2=0.999)
    else:
        raise NotImplementedError

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = Tensor(labels_to_class_weights(dataset.labels, nc) * nc)  # attach class weights
    model.names = names
    if ema:
        ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])

    # Def train func
    compute_loss = ComputeLoss(model)  # init loss class
    if opt.is_distributed:
        mean = context.get_auto_parallel_context("gradients_mean")
        degree = context.get_auto_parallel_context("device_num")
        grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)
    else:
        grad_reducer = ops.functional.identity
    loss_scaler = DynamicLossScaler(2**10, 2, 1000)
    @ms.ms_function
    def forward_func(x, label, sizes=None):
        x /= 255.0
        if sizes is not None:
            x = ops.interpolate(x, sizes=sizes, coordinate_transformation_mode="asymmetric", mode="bilinear")
        pred = model(x)
        loss, loss_items = compute_loss(pred, label)
        return loss_scaler.scale(loss), loss_items

    grad_fn = ops.value_and_grad(forward_func, grad_position=None, weights=optimizer.parameters, has_aux=True)

    all_finite_fn = all_finite if context.get_context("device_target") != "CPU" else all_finite_cpu
    @ms.ms_function
    def train_step(x, label, sizes=None, optimizer_update=True):

        (loss, loss_items), grads = grad_fn(x, label, sizes)
        grads = grad_reducer(grads)
        grads_finite = all_finite_fn(grads)
        unscaled_grads = loss_scaler.unscale(grads)

        if grads_finite:
            if optimizer_update:
                loss = ops.depend(loss, optimizer(unscaled_grads))
                _ = loss_scaler.adjust(grads_finite)
        else:
            print("overflow, loss scale adjust to ", loss_scaler.scale_value)
        return loss, loss_items, unscaled_grads, grads_finite


    # Start training
    data_loader = dataloader.create_dict_iterator(output_numpy=True, num_epochs=300)
    accumulate_grads = None
    accumulate_finite = Tensor(True, ms.bool_)
    model.set_train(True)
    optimizer.set_train(True)
    accumulate_cur_step = 0
    for i, data in enumerate(data_loader):
        s_time = time.time()
        if i < warmup_steps:
            xi = [0, warmup_steps]  # x interp
            accumulate = max(1, np.interp(i, xi, [1, nbs / total_batch_size]).round())
            if opt.optimizer == "sgd":
                optimizer.momentum = momentum_pg[i]
                print("optimizer.momentum: ", optimizer.momentum.asnumpy())

        cur_epoch = (i // per_epoch_size) + 1
        cur_step = (i % per_epoch_size) + 1
        imgs, labels, paths = data["img"], data["label_out"], data["img_files"]
        imgs, labels = Tensor(imgs, ms.float32), Tensor(labels, ms.float32)

        # Multi-scale
        ns = None
        if opt.multi_scale:
            sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                # imgs = ops.interpolate(imgs, sizes=ns, coordinate_transformation_mode="asymmetric", mode="bilinear")

        # Accumulate Grad
        s_train_time = time.time()
        if accumulate == 1:
            _, loss_item, _, _ = train_step(imgs, labels, ns, True)
        else:
            _, loss_item, grads, grads_finite = train_step(imgs, labels, ns, False)
            accumulate_finite = ops.logical_and(accumulate_finite, grads_finite)
            accumulate_cur_step += 1
            if accumulate_grads:
                assert len(accumulate_grads) == len(grads)
                for gi in range(len(grads)):
                    accumulate_grads[gi] += grads[gi]

            else:
                accumulate_grads = list(grads)

            if accumulate_cur_step % accumulate == 0:
                optimizer(tuple(accumulate_grads))
                _ = loss_scaler.adjust(accumulate_finite)
                accumulate_grads = None
                accumulate_cur_step = 0
        _p_train_size = ns if ns else imgs.shape[2:]
        print(f"Epoch {epochs}/{cur_epoch}, Step {per_epoch_size}/{cur_step}, size {_p_train_size}, "
              f"fp/bp time cost: {(time.time() - s_train_time) * 1000:.2f} ms")
        print(f"Epoch {epochs}/{cur_epoch}, Step {per_epoch_size}/{cur_step}, size {_p_train_size}, "
              f"loss: {loss_item[3].asnumpy():.4f}, lbox: {loss_item[0].asnumpy():.4f}, lobj: "
              f"{loss_item[1].asnumpy():.4f}, lcls: {loss_item[2].asnumpy():.4f}, "
              f"step time: {(time.time() - s_time) * 1000:.2f} ms")

        if (rank % 8 == 0) and ((i + 1) % per_epoch_size == 0):
            # Save Checkpoint
            model_name = os.path.basename(opt.cfg)[:-5] # delete ".yaml"
            ckpt_path = os.path.join(wdir, f"{model_name}_{cur_epoch}.ckpt")
            ms.save_checkpoint(model, ckpt_path)
            if ema:
                ema_ckpt_path = os.path.join(wdir, f"EMA_{model_name}_{cur_epoch}.ckpt")
                ms.save_checkpoint(ema.ema_model, ema_ckpt_path, append_dict={"updates": ema.updates})

        # TODO: eval every epoch


if __name__ == '__main__':
    opt = get_args_train()
    # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    opt.name = 'evolve' if opt.evolve else opt.name
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    context.set_context(mode=context.PYNATIVE_MODE, pynative_synchronize=True, device_target=opt.device_target)
    context.set_context(mode=context.GRAPH_MODE, device_target=opt.device_target)
    context.reset_auto_parallel_context()
    # Distribute Train
    rank, rank_size, parallel_mode = 0, 1, ParallelMode.STAND_ALONE
    if opt.is_distributed:
        init()
        rank, rank_size, parallel_mode = get_rank(), get_group_size(), ParallelMode.DATA_PARALLEL
    context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=rank_size)

    opt.total_batch_size = opt.batch_size
    opt.rank_size = rank_size
    opt.rank = rank
    if rank_size > 1:
        assert opt.batch_size % opt.rank_size == 0, '--batch-size must be multiple of device count'
        opt.batch_size = opt.total_batch_size // opt.rank_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Train
    if not opt.evolve:
        # check_train(hyp, opt)
        train(hyp, opt)
    else:
        raise NotImplementedError("Not support evolve train;")