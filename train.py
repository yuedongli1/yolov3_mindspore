# -*- coding: utf-8 -*-

import yaml
import math
import os
import random
import time
import numpy as np
from pathlib import Path

import mindspore as ms
from mindspore import nn, ops, context, Tensor
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size

from network.yolo import Model
from network.common import EMA
from network.loss import ComputeLoss
from config.args import get_args_train
from utils.optimizer import get_group_param_yolov3, get_lr_yolov3
from utils.dataset import create_dataloader
from utils.general import increment_path, colorstr, labels_to_class_weights, check_file, check_img_size


def set_seed(seed=2):
    np.random.seed(seed)
    random.seed(seed)
    ms.set_seed(seed)


def train(hyp, opt):
    save_dir, epochs, batch_size, total_batch_size, rank, freeze = \
        opt.save_dir, opt.epochs, opt.batch_size, opt.total_batch_size, opt.rank, opt.freeze
    set_seed(1+rank)

    if opt.enable_modelarts:
        from utils.modelarts import sync_data
        os.makedirs(opt.data_dir_modelarts, exist_ok=True)
        sync_data(opt.data_url, opt.data_dir_modelarts)

    with open(opt.data, 'rb') as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check
    
    if opt.enable_modelarts:
        data_dir = opt.data_dir_modelarts
    else:
        data_dir = opt.data_dir
    data_dict['train'] = os.path.join(data_dir, data_dict['train'])
    data_dict['val'] = os.path.join(data_dir, data_dict['val'])
    data_dict['test'] = os.path.join(data_dir, data_dict['test'])

    # Directories
    wdir = os.path.join(save_dir, "weights")
    os.makedirs(wdir, exist_ok=True)

    # Save run settings
    with open(os.path.join(save_dir, "hyp.yaml"), 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(os.path.join(save_dir, "opt.yaml"), 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)
    if opt.enable_modelarts:
        sync_data(save_dir, opt.train_url)

    # Model
    sync_bn = opt.sync_bn and context.get_context("device_target") == "Ascend" and rank_size > 1
    model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors'), sync_bn=sync_bn)  # create
    if opt.ema:
        ema = EMA(model)
    else:
        ema = None

    if opt.enable_modelarts:
        pretrained = opt.ckpt_url.endswith('.ckpt')
    else:
        pretrained = opt.weights.endswith('.ckpt')
    if pretrained:
        if opt.enable_modelarts:
            weights_path = '/cache/checkpoint.ckpt'
            sync_data(opt.ckpt_url, weights_path)
        else:
            weights_path = opt.weights
        param_dict = ms.load_checkpoint(weights_path)
        ms.load_param_into_net(model, param_dict)
        print(f"Pretrain model load from \"{weights_path}\" success.")
        if ema:
            if opt.ema_weight.endswith('.ckpt'):
                param_dict_ema = ms.load_checkpoint(opt.ema_weight)
                ms.load_param_into_net(ema.ema_model, param_dict_ema)
                if "updates" in param_dict_ema:
                    ema.updates = param_dict_ema["updates"]
                else:
                    print(f"\"updates\" miss in \"{opt.ema_weight}\"")
                print(f"Ema pretrain model load from \"{opt.ema_weight}\" success.")
            else:
                ema.clone_from_model()
                print("ema_weight not exist, default pretrain weight is currently used.")

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

    train_path = data_dict['train']
    dataloader, dataset, per_epoch_size = create_dataloader(train_path, imgsz, batch_size, gs, opt, epoch_size=opt.epochs,
                                                            hyp=hyp, augment=True, cache=opt.cache_images,
                                                            rect=opt.rect, rank_size=opt.rank_size, rank=opt.rank,
                                                            num_parallel_workers=4 if rank_size > 1 else 8,
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
        if opt.optimizer in ("sgd", "momentum"):
            assert len(momentum_pg) == warmup_steps
        group_params = [{'params': pg0, 'lr': lr_pg0},
                            {'params': pg1, 'lr': lr_pg1, 'weight_decay': hyp['weight_decay']},
                            {'params': pg2, 'lr': lr_pg2}]
    else:
        raise NotImplementedError

    print(f"optimizer loss scale is {opt.ms_optim_loss_scale}")
    if opt.optimizer == "sgd":
        optimizer = nn.SGD(group_params, learning_rate=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    elif opt.optimizer == "adam":
        optimizer = nn.Adam(group_params, learning_rate=hyp['lr0'], beta1=hyp['momentum'], beta2=0.999, loss_scale=opt.ms_optim_loss_scale)
    elif opt.optimizer == "momentum":
        optimizer = nn.Momentum(group_params, learning_rate=hyp['lr0'], momentum=hyp['momentum'], use_nesterov=True,
                                loss_scale=opt.ms_optim_loss_scale)
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

    if opt.is_distributed:
        mean = context.get_auto_parallel_context("gradients_mean")
        degree = context.get_auto_parallel_context("device_num")
        grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)
    else:
        grad_reducer = ops.functional.identity

    from mindspore.amp import StaticLossScaler
    loss_scaler = StaticLossScaler(opt.ms_loss_scaler_value)
    train_step = create_train_static_shape_fn_gradoperation(model, optimizer, loss_scaler, grad_reducer,
                                                                    overflow_still_update=opt.overflow_still_update,
                                                                    sens=opt.ms_grad_sens)

    # Start training
    data_loader = dataloader.create_dict_iterator(output_numpy=True, num_epochs=1)
    s_time_step = time.time()
    s_time_epoch = time.time()
    accumulate_grads = None
    accumulate_cur_step = 0

    model.set_train(True)
    optimizer.set_train(True)

    for i, data in enumerate(data_loader):
        if i < warmup_steps:
            xi = [0, warmup_steps]  # x interp
            accumulate = max(1, np.interp(i, xi, [1, nbs / total_batch_size]).round())
            if opt.optimizer in ("sgd", "momentum"):
                optimizer.momentum = Tensor(momentum_pg[i], ms.float32)
                # print("optimizer.momentum: ", optimizer.momentum.asnumpy())

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
        if accumulate == 1:
            loss, loss_item, _, grads_finite = train_step(imgs, labels, ns, True)
            if ema:
                ema.update()
            if loss_scaler:
                loss_scaler.adjust(grads_finite)
                if not grads_finite:
                    print("overflow, loss scale adjust to ", loss_scaler.scale_value.asnumpy())
        else:
            loss, loss_item, grads, grads_finite = train_step(imgs, labels, ns, False)
            if loss_scaler:
                loss_scaler.adjust(grads_finite)
            if grads_finite:
                accumulate_cur_step += 1
                if accumulate_grads:
                    assert len(accumulate_grads) == len(grads)
                    for gi in range(len(grads)):
                        accumulate_grads[gi] += grads[gi]
                else:
                    accumulate_grads = list(grads)

                if accumulate_cur_step % accumulate == 0:
                    optimizer(tuple(accumulate_grads))
                    if ema:
                        ema.update()
                    print(f"-Epoch: {cur_epoch}, Step: {cur_step}, optimizer an accumulate step success.")
                    # reset accumulate
                    accumulate_grads = None
                    accumulate_cur_step = 0
            else:
                if loss_scaler:
                    print(f"Epoch: {cur_epoch}, Step: {cur_step}, this step grad overflow, drop. "
                              f"Loss scale adjust to {loss_scaler.scale_value.asnumpy()}")
                else:
                    print(f"Epoch: {cur_epoch}, Step: {cur_step}, this step grad overflow, drop. ")

        _p_train_size = imgs.shape[2:]
        print(f"Epoch {epochs}/{cur_epoch}, Step {per_epoch_size}/{cur_step}, size " + str(_p_train_size) + 
                  f"loss: {loss.asnumpy():.4f}, lbox: {loss_item[0].asnumpy():.4f}, lobj: "
                  f"{loss_item[1].asnumpy():.4f}, lcls: {loss_item[2].asnumpy():.4f}, "
                  f"cur_lr: [{lr_pg0[i]:.8f}, {lr_pg1[i]:.8f}, {lr_pg2[i]:.8f}], "
                  f"step time: {(time.time() - s_time_step) * 1000:.2f} ms")

        s_time_step = time.time()

        if (i + 1) % per_epoch_size == 0:
            print(f"Epoch {epochs}/{cur_epoch}, epoch time: {(time.time() - s_time_epoch) / 60:.2f} min.")
            s_time_epoch = time.time()

        if (rank % 8 == 0) and ((i + 1) % per_epoch_size == 0):
            # Save Checkpoint
            model_name = os.path.basename(opt.cfg)[:-5] # delete ".yaml"
            ckpt_path = os.path.join(wdir, f"{model_name}_{cur_epoch}.ckpt")
            ms.save_checkpoint(model, ckpt_path)
            if ema:
                params_list = []
                for p in ema.ema_weights:
                    _param_dict = {'name': p.name[len("ema."):], 'data': Tensor(p.data.asnumpy())}
                    params_list.append(_param_dict)

                ema_ckpt_path = os.path.join(wdir, f"EMA_{model_name}_{cur_epoch}.ckpt")
                ms.save_checkpoint(params_list, ema_ckpt_path, append_dict={"updates": ema.updates})
            if opt.enable_modelarts:
                sync_data(ckpt_path, opt.train_url + "/weights/" + ckpt_path.split("/")[-1])
                if ema:
                    sync_data(ema_ckpt_path, opt.train_url + "/weights/" + ema_ckpt_path.split("/")[-1])

        # TODO: eval every epoch

    return 0


def create_train_static_shape_fn_gradoperation(model, optimizer, loss_scaler, grad_reducer=None, rank_size=8,
                                               overflow_still_update=False, sens=1.0):
    # from mindspore.amp import all_finite # Bugs before MindSpore 1.9.0
    from utils.all_finite import all_finite
    if loss_scaler is None:
        from mindspore.amp import StaticLossScaler
        loss_scaler = StaticLossScaler(sens)
    # Def train func
    compute_loss = ComputeLoss(model)  # init loss class

    if grad_reducer is None:
        grad_reducer = ops.functional.identity

    def forward_func(x, label, sizes=None):
        x /= 255.0
        if sizes is not None:
            x = ops.interpolate(x, sizes=sizes, coordinate_transformation_mode="asymmetric", mode="bilinear")
        pred = model(x)
        loss, loss_items = compute_loss(pred, label)
        loss *= rank_size
        return loss, ops.stop_gradient(loss_items)

    grad_fn = ops.GradOperation(get_by_list=True, sens_param=True)(forward_func, optimizer.parameters)
    sens_value = sens

    @ms.ms_function
    def train_step(x, label, sizes=None, optimizer_update=True):
        loss, loss_items = forward_func(x, label, sizes)
        sens1, sens2 = ops.fill(loss.dtype, loss.shape, sens_value), \
                       ops.fill(loss_items.dtype, loss_items.shape, sens_value)
        grads = grad_fn(x, label, sizes, (sens1, sens2))
        grads = grad_reducer(grads)
        grads = loss_scaler.unscale(grads)
        grads_finite = all_finite(grads)

        if optimizer_update:
            if grads_finite:
                loss = ops.depend(loss, optimizer(grads))
            else:
                if overflow_still_update:
                    loss = ops.depend(loss, optimizer(grads))
                    print("overflow, still update.")
                else:
                    print("overflow, drop the step.")

        return loss, loss_items, grads, grads_finite

    return train_step


if __name__ == '__main__':
    opt = get_args_train()
    if opt.enable_modelarts:
        opt.cfg = os.path.join(opt.file_url, opt.cfg)
        opt.data = os.path.join(opt.file_url, opt.data)
        opt.hyp = os.path.join(opt.file_url, opt.hyp)
    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    opt.name = 'evolve' if opt.evolve else opt.name
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    context.set_context(mode=context.GRAPH_MODE, device_target=opt.device_target)
    # context.set_context(mode=context.PYNATIVE_MODE, device_target=opt.device_target, pynative_synchronize=True)
    if opt.device_target == "Ascend":
        device_id = int(os.getenv('DEVICE_ID', 0))
        context.set_context(device_id=device_id)

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
        train(hyp, opt)
    else:
        raise NotImplementedError("Not support evolve train;")