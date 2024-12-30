import contextlib
import math, config

import torch
import torch.nn as nn
from loguru import logger

from utils.common import fuse_conv_and_bn, intersect_dicts, scale_img, descale_pred
from .common import Conv, Concat, C2f, SPPF, Conv8, SPDConv, WASPConv, ASDownSample, MP2, MP3, C3, ContextGuidedBlock_Down, CrystalResConv, ConvFormer, ContextGuidedBlock, SCDown, C2fCIB, PSA, MP

__all__ = ['YOLOBackbone', 'YOLO8Backbone', 'YOLO10Backbone']

from ...core import register


@register
class YOLOBackbone(nn.Module):
    def __init__(self, cfg, ch=None):
        super().__init__()

        # 根据加载的yaml构建模型
        if isinstance(cfg, dict):
            self.cfg = cfg
        else:
            import yaml
            with open(cfg) as f:
                self.cfg = yaml.load(f, Loader=yaml.SafeLoader)

        if ch is None:
            ch = [config.chs] if isinstance(config.chs, int) else config.chs

        # parse_model
        anchors, nc, gd, gw = self.cfg['anchors'], self.cfg['nc'], self.cfg['depth_multiple'], self.cfg['width_multiple']
        na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # 有多少种不同的anchor，每个anchor size包括宽高
        no = na * (nc + 5)  # anchor数量 * 85， 5是位置和conf

        layers, save, c2 = [], [], ch[-1]  # 网络层, savelist, 输出维度

        for i, (f, n, m, args) in enumerate(self.cfg['backbone']):  # 创建backbone
            m = eval(m) if isinstance(m, str) else m  # m是str或者nn.Module

            # 一层一层的读取网络参数
            for ind, arg in enumerate(args):
                try:
                    args[ind] = eval(arg) if isinstance(arg, str) else arg
                except:
                    pass

            n = max(round(n * gd), 1) if n > 1 else n  # depth gain

            # ------------------------------- 确认参数 ---------------------------------

            # create module in the given list
            if m in [nn.Conv2d, Conv, ASDownSample, MP2, SPDConv, MP3, C3, ConvFormer]:
                c1, c2 = ch[f], args[0]  # 输入输出维度， ch[f]表示通道列表中最后一个数值
                if c2 != no:
                    c2 = math.ceil(c2 * gw / 8) * 8  # 作用将卷积核个数调整到8的整数倍

                args = [c1, c2, *args[1:]]
            elif m is nn.BatchNorm2d:
                args = [ch[f]]
            elif m is Concat:
                c2 = sum([ch[x] for x in f])
            elif m is SPDConv:
                c2 = 2 * ch[f]
            else:
                c2 = ch[f]  # upsample

            # ------------------------------- 确认参数 ---------------------------------

            m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # 创建多层还是一层

            # replace main
            t = str(m)[8:-2].replace('__main__.', '')

            m_.i, m_.f, m_.type = i, f, t  # 当前层索引, 结果来自那一层，长为-1

            # 将这些中间结果保存下来 例如 [-1, -3, -5, -6], 1, Concat, [1]
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)

            layers.append(m_.to(config.device))  # 将m_添加到网络中

            # 第一层的时候，重新创建ch， 然后加入第一层的输出
            if i == 0:
                ch = []
            ch.append(c2)  # 将当前层的输出放在ch列表里，可能作为下一层输入

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        # 初始化模型
        msg = self.load_state_dict(torch.load(config.v7_weights, map_location='cpu'), strict=False)
        logger.warning(f'load v7 backbone {msg}')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]
            f = [None, 3, None]
            y = []
            s = [1, 0.83, 0.67]

            for fi, si in zip(f, s):
                # resize image, 图像预处理  flip是按照维度对输入进行翻转
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]

                yi[..., :4] /= si  # 将坐标还原

                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # 上下
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # 左右
                y.append(yi)
            return torch.cat(y, 1), None
        else:
            return self.forward_once(x)

    def forward_once(self, x):
        y, dt = [], []
        for m in self.model:  # 遍历网络所有层
            if m.f != -1:  # 如果当前层连着不是之前层
                # 当m.f为整数时，取出之前结果y[m.f]  但是当结果为列表list时，m需要的所有之前结果y[j]，并组成一个列表
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

            x = m(x)
            y.append(x if m.i in self.save else None)
            if m.i in [24, 37, 50]:
                dt.append(x)

        return dt

    def _initialize_biases(self, cf=None):  # 初始化detect头的bias参数
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def fuse(self):
        logger.warning('fusing layer ......')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                # 需要初始化
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward

        return self

    def initialize_weights(self):
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
                m.inplace = True


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return self.forward_once(x)

    def forward_once(self, x):
        y, dt = [], []

        for m in self.model:
            if m.f != -1:
                # 如果不是-1就找到对应的输入，这些输入是之前层的输出，被存储在y中
                # 或者，如果m.f是一个数组
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            # if isinstance(x, torch.Tensor):
            #     print(x[0, 0, 0, :3])
            y.append(x if m.i in self.save else None)

        return x

    def is_fused(self, thresh=10):
        # 检测是否已经fuse（如果fuse之后，那么Norm就基本去除了）
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
        return sum(isinstance(v, bn) for v in self.modules()) < thresh

    def _apply(self, fn):
        # 对模型头的参数进行处理
        self = super()._apply(fn)
        return self


@register
class YOLO8Backbone(BaseModel):
    def __init__(self, cfg, ch=None):
        super().__init__()

        # 根据加载的yaml构建模型
        if isinstance(cfg, dict):
            self.cfg = cfg
        else:
            import yaml
            with open(cfg) as f:
                self.cfg = yaml.load(f, Loader=yaml.SafeLoader)

        if ch is None:
            ch = config.chs

        # parse_model
        nc, gd, gw, act = self.cfg['nc'], self.cfg['depth_multiple'], self.cfg['width_multiple'], self.cfg.get('activation')
        if act:
            Conv.default_act = eval(act)

        ch = [ch]

        layers, save, c2 = [], [], ch[-1]  # save list(需要存储的中间结果)， ch out(输出维度)

        for i, (f, n, m, args) in enumerate(self.cfg['backbone']):
            m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]
            for j, a in enumerate(args):
                with contextlib.suppress(NameError):
                    args[j] = eval(a) if isinstance(a, str) else a

            n = max(round(n * gd), 1) if n > 1 else n

            if m in (Conv8, nn.ConvTranspose2d, C2f, SPPF, WASPConv, Conv, SCDown, C2fCIB, PSA):
                c1, c2 = ch[f], args[0]
                if c2 != nc:
                    c2 = math.ceil(c2 * gw / 8) * 8

                args = [c1, c2, *args[1:]]
                if m in [C2f, WASPConv, C2fCIB]:
                    # 如果是c2f要写上重复几次
                    args.insert(2, n)  # number of repeats
                    n = 1
            elif m is nn.BatchNorm2d:
                args = [ch[f]]
            elif m is SPDConv:
                c2 = 2 * ch[f]
            elif m is Concat:
                c2 = sum(ch[x] for x in f)
            elif m in (ContextGuidedBlock_Down, ContextGuidedBlock):
                c2 = ch[f]  # 输出维度
                args = [c2, *args]
            else:
                c2 = ch[f]

            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            t = str(m)[8:-2].replace('__main__.', '')
            m_.i, m_.f, m_.type = i, f, t
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
            layers.append(m_)

            if i == 0:
                ch = []

            ch.append(c2)

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        msg = self.load(torch.load(config.v8_weights, map_location='cpu')['model'])
        logger.warning(f'load v8 backbone {msg}')

    def load(self, weights):
        csd = intersect_dicts(weights, self.state_dict())  # intersect
        return self.load_state_dict(csd, strict=False)  # load

    def warmup(self, imgsz=(1, 3, 640, 640)):
        im = torch.empty(*imgsz, dtype=torch.half if config.half else torch.float, device=config.device)
        for _ in range(1):
            self.forward(im)

    def forward(self, x, augment=False, out_layer_idx=[4, 6, 9]):
        if augment:
            return self._forward_augment(x)
        return self.forward_once(x, out_layer_idx)

    def forward_once(self, x, out_layer_idx):
        y, dt = [], []
        for m in self.model:
            if m.f != -1:
                # 如果不是-1就找到对应的输入，这些输入是之前层的输出，被存储在y中
                # 或者，如果m.f是一个数组
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            # if isinstance(x, torch.Tensor):
            #     print(x[0, 0, 0, :3])
            y.append(x if m.i in self.save else None)
            if m.i in out_layer_idx:  # 469 是根据8的网络模型决定的，并且不同规模的模型这个数值也会产生变化
                dt.append(x)
        # 按照rtdetr的架构，返回应当是backbone的多层结果，这个可能不太对
        return dt

    def fuse(self):
        # fuse conv，将batchnorm和Conv信息融合成
        if not self.is_fused():
            # 对下面三种卷积层进行fuse
            for m in self.model.modules():
                if isinstance(m, Conv) and hasattr(m, 'bn'):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # 在实际使用时，将conv和bn进行fuse，形成一个3X3卷积，加快运行速度
                    delattr(m, 'bn')
                    m.forward = m.fuseforward

        return self

    def is_fused(self, thresh=10):
        # 检测是否已经fuse（如果fuse之后，那么Norm就基本去除了）
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
        return sum(isinstance(v, bn) for v in self.modules()) < thresh

    def _apply(self, fn):
        # 对模型头的参数进行处理
        self = super()._apply(fn)
        m = self.model[-1]  # 取出header
        return self

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # 宽高
        s = [1, 0.83, 0.67]  # 缩放系数
        f = [None, 3, None]  # 图像反转 (2-上下反转, 3-左右反转)
        y = []
        for si, fi in zip(s, f):
            # 进行图像缩放
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]
            # 还原scale，将结果还原到原始图像坐标
            yi = descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self.clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None

    def clip_augmented(self, y):
        # 裁剪数据增强
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y


@register
class YOLO10Backbone(BaseModel):
    def __init__(self, cfg, ch=None):
        super().__init__()

        # 根据加载的yaml构建模型
        if isinstance(cfg, dict):
            self.cfg = cfg
        else:
            import yaml
            with open(cfg) as f:
                self.cfg = yaml.load(f, Loader=yaml.SafeLoader)

        if ch is None:
            ch = config.chs

        # parse_model
        nc, gd, gw, act = self.cfg['nc'], self.cfg['depth_multiple'], self.cfg['width_multiple'], self.cfg.get('activation')
        if act:
            Conv.default_act = eval(act)

        ch = [ch]

        layers, save, c2 = [], [], ch[-1]  # save list(需要存储的中间结果)， ch out(输出维度)

        for i, (f, n, m, args) in enumerate(self.cfg['backbone']):
            m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]
            for j, a in enumerate(args):
                with contextlib.suppress(NameError):
                    args[j] = eval(a) if isinstance(a, str) else a

            n = max(round(n * gd), 1) if n > 1 else n

            if m in (Conv8, nn.ConvTranspose2d, C2f, SPPF, WASPConv, Conv, SCDown, C2fCIB, PSA):
                c1, c2 = ch[f], args[0]
                if c2 != nc:
                    c2 = math.ceil(c2 * gw / 8) * 8

                args = [c1, c2, *args[1:]]
                if m in [C2f, WASPConv, C2fCIB]:
                    # 如果是c2f要写上重复几次
                    args.insert(2, n)  # number of repeats
                    n = 1
            elif m is nn.BatchNorm2d:
                args = [ch[f]]
            elif m is SPDConv:
                c2 = 2 * ch[f]
            elif m is Concat:
                c2 = sum(ch[x] for x in f)
            elif m in (ContextGuidedBlock_Down, ContextGuidedBlock):
                c2 = ch[f]  # 输出维度
                args = [c2, *args]
            else:
                c2 = ch[f]

            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            t = str(m)[8:-2].replace('__main__.', '')
            m_.i, m_.f, m_.type = i, f, t
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
            layers.append(m_)

            if i == 0:
                ch = []

            ch.append(c2)

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        msg = self.load(torch.load(config.v10_weights, map_location='cpu'))
        logger.warning(f'load v8 backbone {msg}')

    def load(self, weights):
        csd = intersect_dicts(weights, self.state_dict())  # intersect
        return self.load_state_dict(csd, strict=False)  # load

    def warmup(self, imgsz=(1, 3, 640, 640)):
        im = torch.empty(*imgsz, dtype=torch.half if config.half else torch.float, device=config.device)
        for _ in range(1):
            self.forward(im)

    def forward(self, x, augment=False, out_layer_idx=[4, 6, 9]):
        if augment:
            return self._forward_augment(x)
        return self.forward_once(x, out_layer_idx)

    def forward_once(self, x, out_layer_idx):
        y, dt = [], []
        for m in self.model:
            if m.f != -1:
                # 如果不是-1就找到对应的输入，这些输入是之前层的输出，被存储在y中
                # 或者，如果m.f是一个数组
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            # if isinstance(x, torch.Tensor):
            #     print(x[0, 0, 0, :3])
            y.append(x if m.i in self.save else None)
            if m.i in out_layer_idx:  # 469 是根据8的网络模型决定的，并且不同规模的模型这个数值也会产生变化
                dt.append(x)
        # 按照rtdetr的架构，返回应当是backbone的多层结果，这个可能不太对
        return dt

    def fuse(self):
        # fuse conv，将batchnorm和Conv信息融合成
        if not self.is_fused():
            # 对下面三种卷积层进行fuse
            for m in self.model.modules():
                if isinstance(m, Conv) and hasattr(m, 'bn'):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # 在实际使用时，将conv和bn进行fuse，形成一个3X3卷积，加快运行速度
                    delattr(m, 'bn')
                    m.forward = m.fuseforward

        return self

    def is_fused(self, thresh=10):
        # 检测是否已经fuse（如果fuse之后，那么Norm就基本去除了）
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
        return sum(isinstance(v, bn) for v in self.modules()) < thresh

    def _apply(self, fn):
        # 对模型头的参数进行处理
        self = super()._apply(fn)
        m = self.model[-1]  # 取出header
        return self

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # 宽高
        s = [1, 0.83, 0.67]  # 缩放系数
        f = [None, 3, None]  # 图像反转 (2-上下反转, 3-左右反转)
        y = []
        for si, fi in zip(s, f):
            # 进行图像缩放
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]
            # 还原scale，将结果还原到原始图像坐标
            yi = descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self.clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None

    def clip_augmented(self, y):
        # 裁剪数据增强
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y