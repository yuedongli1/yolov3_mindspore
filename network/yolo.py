import math
import numpy as np
from pathlib import Path
from copy import deepcopy

import mindspore as ms
from mindspore import nn, ops, Tensor
from .common import parse_model, Detect
from utils.autoanchor import check_anchor_order

import os, sys
# sys.path.insert(0, "/disk3/zhy/_YOLO/yolo_mindspore")
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)


def initialize_weights(model):
    for m in model.cells():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.97


@ops.constexpr
def _get_h_w_list(ratio, gs, hw):
    return tuple([math.ceil(x * ratio / gs) * gs for x in hw])


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = ops.ResizeBilinear(size=s, align_corners=False)(img)
        if not same_shape:  # pad/crop img
            h, w = _get_h_w_list(ratio, gs, (h, w))

        # img = F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean
        img = ops.pad(img, ((0, 0), (0, 0), (0, w - s[1]), (0, h - s[0])))
        img[:, :, -(w - s[1]):, :] = 0.447
        img[:, :, :, -(h - s[0]):] = 0.447
        return img


@ops.constexpr
def _get_stride_max(stride):
    return int(stride.max())


class Model(nn.Cell):
    def __init__(self, cfg='yolov3.yaml', ch=3, nc=None, anchors=None, sync_bn=False):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            print(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            print(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save, self.layers_param = parse_model(deepcopy(self.yaml), ch=[ch])
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.stride = Tensor(np.array(self.yaml['stride']), ms.int32)
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self.stride_np = np.array(self.yaml['stride'])
            self._initialize_biases()  # only run once

    def construct(self, x, augment=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = (1, 0.83, 0.67)  # scales
            f = (None, 3, None)  # flips (2-ud, 3-lr)
            y = ()  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(ops.ReverseV2(fi)(x) if fi else x, si, gs=_get_stride_max(self.stride_np))
                yi = self.forward_once(xi)[0]  # forward
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y += (yi,)
            return ops.concat(y, 1)  # augmented inference, train
        else:
            return self.forward_once(x)  # single-scale inference, train

    def forward_once(self, x):
        y, dt = (), ()  # outputs
        for i in range(len(self.model)):
            m = self.model[i]
            iol, f, _, _ = self.layers_param[i]  # iol: index of layers

            if not(isinstance(f, int) and f == -1): # if not from previous layer
                # x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                if isinstance(f, int):
                    x = y[f]
                else:
                    _x = ()
                    for j in f:
                        if j == -1:
                            _x += (x,)
                        else:
                            _x += (y[j],)
                    x = _x

            # print("index m: ", iol) # print if debug on pynative mode, not available on graph mode.
            x = m(x)  # run

            y += (x if iol in self.save else None,)  # save output

        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            s = s.asnumpy()
            b = mi.bias.view(m.na, -1).asnumpy()  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else np.log(cf / cf.sum())  # cls
            mi.bias = ops.assign(mi.bias, Tensor(b, ms.float32).view(-1))


if __name__ == '__main__':
    from mindspore import context
    from config.args import get_args
    from utils.general import check_file, increment_path, colorstr
    import yaml
    from utils.dataset import create_dataloader

    context.set_context(mode=context.GRAPH_MODE, pynative_synchronize=True)
    context.set_context(mode=context.PYNATIVE_MODE, pynative_synchronize=True)
    cfg = "D:/yolov3_mindspore/config/network/yolov3.yaml"
    model = Model(cfg, ch=3, nc=80, anchors=None)
    for p in model.trainable_params():
        print(p.name)
    model.set_train(True)
    # x = Tensor(np.random.randint(0, 256, (1, 3, 160, 160)), ms.float32)
    # pred = model(x)
    # print(pred)
    opt = get_args()
    # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    opt.name = 'evolve' if opt.evolve else opt.name
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run
    opt.total_batch_size = opt.batch_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Train set
    train_path = "D:/yolov3_fromv7/datasets/coco/train2017.txt"
    train_path = "D:/yolov3_fromv7/datasets/coco128"
    imgsz, _ = [640, 640]
    batch_size = 2
    gs = 32
    ds, _, ds_size = create_dataloader(train_path, imgsz, batch_size, gs, opt, epoch_size=1,
                                       hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect,
                                       rank=0, rank_size=1,
                                       num_parallel_workers=8, image_weights=opt.image_weights,
                                       quad=opt.quad, prefix=colorstr('train: '))
    data_loader = ds.create_dict_iterator(output_numpy=True, num_epochs=1)
    max_shape = 0
    for i, data in enumerate(data_loader):
        imgs, label_outs, img_files = data["img"], data["label_out"], data["img_files"]
        print(f"{i}/{ds_size} label_outs: {label_outs.shape}")
        if label_outs.shape[1] > max_shape:
            max_shape = label_outs.shape[1]
            print(f"max shape: {max_shape}")
        pred = model(imgs)
        print('niubi')