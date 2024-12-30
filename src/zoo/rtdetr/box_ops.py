'''
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/util/box_ops.py
'''

import torch
from torchvision.ops.boxes import box_area
import torch.nn.functional as F


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def NULLIoU(boxes1, boxes2, alpha=1):
    """
    一种新的IoU计算方式，轻量化计算方法，加快训练速度
    boxes都是[x0, y0, x1, y1]格式的
    output, target  预测框 真实框
    我们认为预测框和真实框之间的损失来自于两个部分 一部分是距离，另一部分是形状大小
    """
    # 防泄漏验证
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()  # degenerate boxes gives inf / nan results
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    # 宽高比例
    wh1 = boxes1[:, None, :2] - boxes1[:, None, 2:]
    wh2 = boxes2[:, None, :2] - boxes2[:, None, 2:]
    ratios = wh1 / wh2  # 预测框/真实框  (0, 1000]
    ratio = torch.exp(torch.pow(torch.arctan(ratios - 1) * 2 / torch.pi, 2)) - 1  # 让曲线更加平缓
    ratio = ratio.sum(-1)

    # 中心距离/最小矩形对角线距离
    lt = torch.min(boxes1[:, None, :2], boxes2[:, None, :2])  # [N,M,2]
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, None, 2:])  # [N,M,2]
    # dis 描述两个框之间的距离，取值在[0,1)之间, 或者为了计算复杂度考虑，可以用（rb[0]-lt[0]）代替
    dis = torch.sqrt(((boxes1[:, None, 0] + boxes1[:, None, 2]) / 2 - (boxes2[:, None, 0] + boxes2[:, None, 2]) / 2) ** 2
                     + ((boxes1[:, None, 1] + boxes1[:, None, 3]) / 2 - (boxes2[:, None, 1] + boxes2[:, None, 3]) / 2) ** 2) \
          / torch.sqrt((lt[:, :, 0] - rb[:, :, 0]) ** 2 + (lt[:, :, 1] - rb[:, :, 1]) ** 2)

    mpdiou = dis + ratio
    mpdiou = mpdiou.squeeze(-1)
    return mpdiou


def MPDIoU(boxes1, boxes2):
    """
    一种新的IoU计算方式，可以替代GIoU
    boxes都是[x0, y0, x1, y1]格式的
    """

    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()  # degenerate boxes gives inf / nan results
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)  # 计算得到IOU

    d1 = (boxes2[:, 0] - boxes1[:, 0]) ** 2 + (boxes2[:, 1] - boxes1[:, 1]) ** 2
    d2 = (boxes2[:, 2] - boxes1[:, 2]) ** 2 + (boxes2[:, 3] - boxes1[:, 3]) ** 2

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)

    # 我们使用 factor 代替图像wh
    factor = wh[:, :, 0] ** 2 + wh[:, :, 1] ** 2
    mpdiou = iou - d1 / factor - d2 / factor
    return mpdiou


def box_iou_loss(boxes1, boxes2):
    # boxes1必须是target boxes
    # boxes2必须是pred boxes
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    union = area1[:, None] + area2 - inter
    # 对于那些重合面积大于预测框85%的, 并且与目标框重合小于0.5的box，我们认为它是干扰box，需要进行去除
    loss_iou = union[torch.logical_and(torch.logical_and(inter / area2.unsqueeze(0) > 0.85, inter / area1.unsqueeze(1) < 0.5), 0.01 < inter / area1.unsqueeze(1))]

    return loss_iou.sum() / len(loss_iou)


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        print(boxes1)
        raise ValueError('error !')
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        print(boxes2)
        raise ValueError('error !')
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
