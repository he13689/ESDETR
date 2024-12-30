'''by lyuwenyu
'''

import copy
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import config
from .utils import get_activation

from src.core import register

__all__ = ['HybridEncoder', 'HybridEncoder2', 'HybridEncoder3', 'HybridEncoder4']

from ...nn import TranCFormer, DeformConv2d, Conv


def tensor_to_heatmap(tensor, colormap='jet'):
    tensor = tensor.detach().cpu().numpy()  # 转换为 Numpy
    summed_tensor = np.sum(tensor, axis=0)  # 在通道维度上求和，生成 (H, W) 大小的数组

    # 将值归一化到 [0, 1]
    normalized_tensor = (summed_tensor - np.min(summed_tensor)) / (np.max(summed_tensor) - np.min(summed_tensor) + 1e-8)
    cmap = plt.get_cmap(colormap)
    heatmap = cmap(normalized_tensor)
    heatmap = np.uint8(heatmap[:, :, :3] * 255)  # 转换为 RGB 并缩放到 0-255
    return heatmap


# 将多个张量转换为热力图列表
def convert_tensors_to_heatmaps(tensor, target_size=(80, 80)):
    heatmap = tensor_to_heatmap(tensor[0])  # 转换为 RGB 图像
    heatmap_resized = cv2.resize(heatmap, target_size)  # 调整大小到目标尺寸
    return heatmap_resized


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        # self.__delattr__('conv1')
        # self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        # 多头注意力机制 nhead一般为8
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout) if not config.soft_dropout else SoftDropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout) if not config.soft_dropout else SoftDropout(dropout)
        self.dropout2 = nn.Dropout(dropout) if not config.soft_dropout else SoftDropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        tensor = tensor.half() if config.half else tensor
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src  # 残差
        if self.normalize_before:  # 正则化在前或者在后
            src = self.norm1(src)

        q = k = self.with_pos_embed(src, pos_embed)  # 是否加入位置编码
        q = q.half() if config.half else q
        k = k.half() if config.half else k
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)  # 将qkv传入进行attn计算，

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class ConvAttentionLayer(nn.Module):
    def __init__(self, d_model,
                 nhead,
                 dropout=0.1):
        super().__init__()
        self.dmodel = d_model
        self.qconv = Conv(20, 20, 3, 1, 1)
        self.kconv = Conv(20, 20, 3, 1, 1)
        self.vconv = Conv(d_model, d_model, 3, 1, 1)
        self.main = Conv(d_model, d_model, 1, 1, 0)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

    def forward(self, x, attn_mask=None):
        # q: 4, 256, 20, 20 -> 4, 20, 256, 20
        # k: 4, 256, 20, 20 -> 4, 20, 20, 256
        # v: 4, 256, 20, 20

        q = self.qconv(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3).flatten(2).transpose(1, 2)
        k = self.kconv(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1).flatten(2).transpose(1, 2)
        v = self.vconv(x).flatten(2).transpose(1, 2)
        y, _ = self.self_attn(q, k, value=v, attn_mask=attn_mask)
        return y


class SoftDropout(torch.nn.Module):
    def __init__(self, p=0.5, softhat=config.softhat):
        super(SoftDropout, self).__init__()
        self.softhat = softhat
        self.p = p  # Dropout概率

    def forward(self, x):
        if self.training:  # 只在训练阶段应用Dropout
            mask = torch.ones_like(x)
            dropout_mask = (torch.rand_like(x) > self.p).float()
            mask *= dropout_mask

            normal_values = torch.clamp(torch.randn_like(x) * self.softhat, min=0, max=self.softhat)
            mask[mask == 0] = normal_values[mask == 0]
            y = x * mask / (1 - self.p)

            return y  # 应用mask，并且对未被置零的元素进行缩放
        else:
            # 在评估模式下直接返回输入
            return x


class EnCover(nn.Module):
    '''
    不需要进行拉直，在这里在进行拉直即可
    input is bsx256x20x20
    '''

    def __init__(self, d_model, nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = ConvAttentionLayer(d_model, nhead, dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.BatchNorm2d(d_model) if normalize_before else nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    def forward(self, x, src_mask=None):
        residual = x

        if self.normalize_before:  # 正则化在前或者在后
            x = self.norm1(x)

        x = self.self_attn(x, attn_mask=src_mask)

        x = residual.flatten(2).transpose(1, 2) + self.dropout1(x)

        if not self.normalize_before:
            x = self.norm1(x)

        # 线性层
        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = residual + self.dropout2(x)
        if not self.normalize_before:
            x = self.norm2(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


@register
class HybridEncoder(nn.Module):
    """
    this is efficient hybrid encoder. 一层 Transformer 的 Encoder，相当于颈部网络
    包括两个部分 AIFI和CCFM
    """

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size

        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    # 这里hidden_dim是统一的，都是256，可以考虑增加
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim)
                )
            )

        # encoder transformer， 这个是encoder
        if not config.new_encoder:  # 新的encoder
            encoder_layer = TransformerEncoderLayer(
                hidden_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=enc_act)
        else:
            encoder_layer = EnCover(  # 新的encoder
                hidden_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=enc_act)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        '''
        '''
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        # proj_feats的主要作用就是将特征维度进行缩放，变成相同的
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # # 将 80x80, 40x40 和 20x20 Tensor 转换为热力图并统一到 80x80, 用于注意力可视化
        # heatmaps_80x80 = convert_tensors_to_heatmaps(proj_feats[0], target_size=(640, 640))
        # heatmaps_40x40 = convert_tensors_to_heatmaps(proj_feats[1], target_size=(640, 640))
        # heatmaps_20x20 = convert_tensors_to_heatmaps(proj_feats[2], target_size=(640, 640))
        #
        # # 保存图像到 PNG 文件
        # cv2.imwrite(f'{config.image_list_dir}/final_heatmap_grid{config.counter}-80.png', heatmaps_80x80)
        # cv2.imwrite(f'{config.image_list_dir}/final_heatmap_grid{config.counter}-40.png', heatmaps_40x40)
        # cv2.imwrite(f'{config.image_list_dir}/final_heatmap_grid{config.counter}-20.png', heatmaps_20x20)

        # background_img = cv2.imread(config.image_list_dir+config.image_list_src[config.counter])
        # background_img = cv2.resize(background_img, (640, 640))
        # # blended_img = cv2.addWeighted(cv2.cvtColor(heatmaps_80x80, cv2.COLOR_BGR2RGB), 0.5, background_img, 0.5, 0)  # 混合
        # blended_img = cv2.addWeighted(heatmaps_80x80, 0.5, background_img, 0.5, 0)  # 混合
        # cv2.imwrite(f'{config.image_list_dir}/final_blend_{config.counter}-80.png', blended_img)
        #
        # # blended_img = cv2.addWeighted(cv2.cvtColor(heatmaps_40x40, cv2.COLOR_BGR2RGB), 0.5, background_img, 0.5, 0)  # 混合
        # blended_img = cv2.addWeighted(heatmaps_40x40, 0.5, background_img, 0.5, 0)  # 混合
        # cv2.imwrite(f'{config.image_list_dir}/final_blend_{config.counter}-40.png', blended_img)
        #
        # # blended_img = cv2.addWeighted(cv2.cvtColor(heatmaps_20x20, cv2.COLOR_BGR2RGB), 0.5, background_img, 0.5, 0)  # 混合
        # blended_img = cv2.addWeighted(heatmaps_20x20, 0.5, background_img, 0.5, 0)  # 混合
        # cv2.imwrite(f'{config.image_list_dir}/final_blend_{config.counter}-20.png', blended_img)
        #
        # config.counter += 1

        # encoder
        if self.num_encoder_layers > 0:
            # self.use_encoder_idx = [2]  所以只要第二维，也就是要第三个输出
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]  # 经过
                # flatten [B, C, H, W] to [B, HxW, C]  在这里j
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)
                # 只有一个encoder，接收第三个输出，并且要加入位置编码
                if config.new_encoder:  # 是否使用encover
                    memory = self.encoder[i](proj_feats[enc_ind])  # 输入卷积而不是拉直后的向量
                else:
                    memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # broadcasting and fusion  扩散和融合，这一步就是CCFM，所以根本来说，就是FPN+PAN
        inner_outs = [proj_feats[-1]]
        # CCFM中红色线
        for idx in range(len(self.in_channels) - 1, 0, -1):
            # 将当前特征图22X22和上一级特征图44X44融合起来
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)  # 将当前特征图经过lateral_convs，都是1X1卷积核，不改变特征图大小
            inner_outs[0] = feat_heigh
            # 将当前特征图上采样，对应结构图中CCFM的fusion
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            # fusion就是上采样+fpn block，所以这个fpn block也是fusion一部分
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        # CCFM中蓝色线
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            outs.append(out)  # 这个outs中存放的就是三个黑线中的结果

        return outs


@register
class HybridEncoder2(nn.Module):
    """
    去掉FPN，直接接PAN, 节省网络空间
    """

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size

        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    # 这里hidden_dim是统一的，都是256，可以考虑增加
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim)
                )
            )

        # encoder transformer， 这个是encoder
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion))

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        '''
        '''
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        # proj_feats的主要作用就是将特征维度进行缩放，变成相同的
        # feats  256 8080, 512 4040, 512 2020 -> 256 将特征通道数转为256
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # encoder
        if self.num_encoder_layers > 0:  # 只有一层，所以就是1
            # self.use_encoder_idx = [2]  所以只要第二维，也就是要第三个输出
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]  # 经过
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)
                # 只有一个encoder，接收第三个输出，并且要加入位置编码
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        inner_outs = [proj_feats[-1]]
        # CCFM中红色线
        for idx in range(len(self.in_channels) - 1, 0, -1):
            # 将当前特征图22X22和上一级特征图44X44融合起来
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)  # 将当前特征图经过lateral_convs，都是1X1卷积核，不改变特征图大小
            inner_outs[0] = feat_heigh
            # 将当前特征图上采样，对应结构图中CCFM的fusion
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            # fusion就是上采样+fpn block，所以这个fpn block也是fusion一部分
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        # 256 8080, 256 4040, 256 2020
        return inner_outs


@register
class HybridEncoder3(nn.Module):
    """
    this is efficient hybrid encoder. 一层 Transformer 的 Encoder，相当于颈部网络
    包括两个部分 AIFI和CCFM
    """

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size

        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                TranCFormer(in_channel, hidden_dim)
            )

        # encoder transformer， 这个是encoder
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion))

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        '''
        '''
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        # proj_feats的主要作用就是将特征维度进行缩放，变成相同的
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # encoder
        if self.num_encoder_layers > 0:
            # self.use_encoder_idx = [2]  所以只要第二维，也就是要第三个输出
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]  # 经过
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)
                # 只有一个encoder，接收第三个输出，并且要加入位置编码
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # broadcasting and fusion  扩散和融合，这一步就是CCFM，所以根本来说，就是FPN+PAN
        inner_outs = [proj_feats[-1]]
        # CCFM中红色线
        for idx in range(len(self.in_channels) - 1, 0, -1):
            # 将当前特征图22X22和上一级特征图44X44融合起来
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)  # 将当前特征图经过lateral_convs，都是1X1卷积核，不改变特征图大小
            inner_outs[0] = feat_heigh
            # 将当前特征图上采样，对应结构图中CCFM的fusion
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            # fusion就是上采样+fpn block，所以这个fpn block也是fusion一部分
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        # CCFM中蓝色线
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            outs.append(out)  # 这个outs中存放的就是三个黑线中的结果

        return outs


@register
class HybridEncoder4(nn.Module):
    """
    this is HFE + TranCFormer
    """

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size

        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                TranCFormer(in_channel, hidden_dim)
            )

        # encoder transformer， 这个是encoder
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        '''
        '''
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        # proj_feats的主要作用就是将特征维度进行缩放，变成相同的
        # feats  256 8080, 512 4040, 512 2020 -> 256 将特征通道数转为256
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # encoder
        if self.num_encoder_layers > 0:  # 只有一层，所以就是1
            # self.use_encoder_idx = [2]  所以只要第二维，也就是要第三个输出
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]  # 经过
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)
                # 只有一个encoder，接收第三个输出，并且要加入位置编码
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        inner_outs = [proj_feats[-1]]
        # CCFM中红色线
        for idx in range(len(self.in_channels) - 1, 0, -1):
            # 将当前特征图22X22和上一级特征图44X44融合起来
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)  # 将当前特征图经过lateral_convs，都是1X1卷积核，不改变特征图大小
            inner_outs[0] = feat_heigh
            # 将当前特征图上采样，对应结构图中CCFM的fusion
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            # fusion就是上采样+fpn block，所以这个fpn block也是fusion一部分
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        # 256 8080, 256 4040, 256 2020
        return inner_outs
