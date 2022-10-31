import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor
from utils.general import make_divisible
import math
import mindspore.numpy as mnp
from mindspore.common.initializer import HeUniform


_SYNC_BN = False


class ResizeNearestNeighbor(nn.Cell):
    def __init__(self, scale=2):
        super(ResizeNearestNeighbor, self).__init__()
        self.scale = scale

    def construct(self, x):
        return ops.ResizeNearestNeighbor((x.shape[-2] * 2, x.shape[-1] * 2))(x)


def _calculate_fan_in_and_fan_out(shape):
    dimensions = len(shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = shape[1]
    num_output_fmaps = shape[0]
    receptive_field_size = 1
    if dimensions > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def _init_bias(conv_weight_shape):
    bias_init = None
    fan_in, _ = _calculate_fan_in_and_fan_out(conv_weight_shape)
    if fan_in != 0:
        bound = math.sqrt(1 / fan_in)
        bias_init = Tensor(np.random.uniform(-bound, bound, conv_weight_shape[0]), dtype=ms.float32)
    return bias_init


def _init_weights(conv_weight_shape):
    weights_init = None
    fan_in, _ = _calculate_fan_in_and_fan_out(conv_weight_shape)
    if fan_in != 0:
        bound = math.sqrt(1 / fan_in)
        weights_init = Tensor(np.random.uniform(-bound, bound, conv_weight_shape), dtype=ms.float32)
    return weights_init


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Concat(nn.Cell):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def construct(self, x):
        return ops.concat(x, self.d)


class Conv(nn.Cell):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s,
                              pad_mode="pad",
                              padding=autopad(k, p),
                              group=g,
                              has_bias=False,
                              # weight_init=HeUniform(negative_slope=5))
                              weight_init=_init_weights((c2, c1, k, k)))
        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Cell) else nn.Identity())

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
        # return self.act(self.bn(self.conv(x)))


class BaseCell(nn.Cell):
    def __init__(self, parameter):
        super(BaseCell, self).__init__()
        self.param = parameter


class Bottleneck(nn.Cell):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def construct(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Detect(nn.Cell):
    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.stride = None

        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        # self.grid = [Tensor(np.zeros(1), ms.float32)] * self.nl
        self.grid = nn.CellList([BaseCell(ms.Parameter(Tensor(np.zeros(1), ms.float32), requires_grad=False)) for _ in range(self.nl)])
        # self.anchor_grid = [Tensor(np.zeros(1), ms.float32)] * self.nl
        self.anchor_grid = nn.CellList(
            [BaseCell(ms.Parameter(Tensor(np.zeros(1), ms.float32), requires_grad=False)) for _ in range(self.nl)])
        self.anchors = Tensor(anchors, ms.float32).view(self.nl, -1, 2)

        self.m = nn.CellList([nn.Conv2d(x, self.no * self.na, 1,
                                        pad_mode="valid",
                                        has_bias=True, weight_init=HeUniform(negative_slope=5), bias_init=_init_bias((self.no * self.na, x, 1, 1))) for x in ch])  # output conv
        # self.inplace = inplace

    @ms.ms_function
    def construct(self, x):
        # x = x.copy()  # for profiling
        z = ()  # inference output

        outs = ()
        for i in range(self.nl):
            out = self.m[i](x[i])  # conv
            bs, _, ny, nx = out.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            out = ops.Transpose()(out.view(bs, self.na, self.no, ny, nx), (0, 1, 3, 4, 2))
            outs += (out,)

            # if not self.training:  # inference
            #     grid_i_shape = self.grid[i].param.shape
            #     x_i_shape = x[i].shape
            #     if len(grid_i_shape) < 4 or grid_i_shape[2] != x_i_shape[2] or grid_i_shape[3] != x_i_shape[3]:
            #     # if self.grid[i].shape[2:4] != x[i].shape[2:4]:
            #     #     self.grid_cell[i].param = self._make_grid(nx, ny)
            #         tmp = self._make_grid(nx, ny, i)
            #         self.grid[i].param = ops.assign(self.grid[i].param, tmp[0])
            #         self.anchor_grid[i].param = ops.assign(self.anchor_grid[i].param, tmp[1])
            #
            #     y = ops.Sigmoid()(x[i])
            #     y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].param) * self.stride[i]  # xy
            #     y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].param  # wh
            #     z += (y.view(bs, -1, self.no),)

        return outs
        # return outs if self.training else (ops.concat(z, 1), outs)

    def _make_grid(self, nx=20, ny=20, i=0):
        yv, xv = ops.meshgrid((mnp.arange(ny), mnp.arange(nx)))
        grid = ops.cast(ops.broadcast_to(ops.stack((xv, yv), 2), (1, self.na, ny, nx, 2)), ms.float32)
        anchor_grid = ops.cast(ops.broadcast_to((self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)), (1, self.na, ny, nx, 2)), ms.float32)
        return grid, anchor_grid


def parse_model(d, ch, sync_bn=False):  # model_dict, input_channels(3)
    _SYNC_BN = sync_bn
    print('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    layers_param = []
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, Bottleneck]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
        elif m in (nn.BatchNorm2d, nn.SyncBatchNorm):
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m in [Detect, ]:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        m_ = nn.SequentialCell([m(*args) for _ in range(n)]) if n > 1 else m(*args)

        t = str(m) # module type
        np = sum([x.size for x in m_.get_parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        layers_param.append((i, f, t, np))
        print('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.CellList(layers), sorted(save), layers_param


class ModelEMA:

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        model.set_train(False)
        self.ema_model = model
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)

    def update(self, model):
        # Update EMA parameters
        self.updates += 1
        d = self.decay(self.updates)

        train_parameters = model.parameters_dict()
        for k, v in self.ema_model.parameters_dict():
            v_np = v.asnumpy()
            v_np *= d
            v_np += (1. - d) * train_parameters[k].asnumpy()
            v.set_data(Tensor(v_np, v.dtype))

    def copy_attr(self, a, b, include=(), exclude=()):
        # Copy attributes from b to a, options to only include [...] and to exclude [...]
        for k, v in b.__dict__.items():
            if (len(include) and k not in include) or k.startswith('_') or k in exclude:
                continue
            else:
                setattr(a, k, v)

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        self.copy_attr(self.ema_model, model, include, exclude)