import math

import numpy as np
import torch.nn.functional as F
import torch
from torch import nn

INCLUDE_KEY = '__include__'


def merge_dict(dct, another_dct):
    '''merge another_dct into dct
    '''
    for k in another_dct:
        if (k in dct and isinstance(dct[k], dict) and isinstance(another_dct[k], dict)):
            merge_dict(dct[k], another_dct[k])
        else:
            dct[k] = another_dct[k]

    return dct


def fuse_conv_and_bn(conv, bn):
    fusedconv = nn.Conv2d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride,
                          padding=conv.padding, groups=conv.groups, bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    if ratio == 1.0:  # 不进行缩放
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)
        if not same_shape:
            # h,w 必须是gs的整数倍
            h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
        # [0, w - s[1], 0, h - s[0]] 表示 (左边填充数， 右边填充数， 上边填充数， 下边填充数)
        # 即在img 的右边和下边添加值为value的区域，大小如上
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)


def descale_pred(p, flips, scale, img_size, dim=1):
    # 逆转scale操作
    p[:, :4] /= scale  # de-scale
    x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
    if flips == 2:
        y = img_size[0] - y  # de-flip ud
    elif flips == 3:
        x = img_size[1] - x  # de-flip lr
    return torch.cat((x, y, wh, cls), dim)


def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0:
        r, g, b = v, t, p
    elif hi == 1:
        r, g, b = q, v, p
    elif hi == 2:
        r, g, b = p, v, t
    elif hi == 3:
        r, g, b = p, q, v
    elif hi == 4:
        r, g, b = t, p, v
    elif hi == 5:
        r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b


def generate_colors(classes):
    assert isinstance(classes, int), 'classes should be int'
    # classes + 1 防止首尾颜色一样
    hs = np.linspace(0, 360, classes + 1, dtype=float)
    hsvs = [[h, 1, 1] for h in hs]
    rgbs = [hsv2rgb(*hsv) for hsv in hsvs]

    rgb_array = np.array(rgbs)
    rgb_array = rgb_array[:, ::-1]  # rgb顺序改为bgr，因为cv2的顺序不一样
    return rgb_array