import os

from utils.common import generate_colors

model_yaml = 'configs/rtdetr/rtdetr_r101vd_6x_coco.yml'  # 模型yaml配置文件
resume = False  # 继续训练
use_amp = True  # amp加速
tuning = 'weights/rtdetr_r101vd_6x_coco_from_paddle.pth'  # 预训练模型位置

conf_thres = .32  # 置信度筛选
new_loss = False
discount_factor = .5  # 加权box损失

test_only = False  # 是否推理
chs = 3  # 输入通道数
device = 'cuda'
half = False  # 半精度
v8_weights = 'weights/yolom.pt'
v7_weights = 'weights/yolov7.pth'
v10_weights = 'weights/yolov10_l.pt'
save_counter = 0
colors = generate_colors(55)  # 根据类别选择颜色数量
best_ap = 0.300
counter = 0
image_list_dir = 'attention_map_test/rsod/'
image_list_src = os.listdir(image_list_dir)

softhat = 0.05  # for sd, default is 0.05
mask_percent = 0.025  # for gna, default is 10
noise_intense = 0.1  # for gna, default is 0.1

new_method = True
if new_method:
    new_encoder = True
    iou_type = 'MPDIOU'  # NIOU MPDIOU GIOU, MPDIOU效果更好
    soft_dropout = True
    mask_image = True
else:
    new_encoder = False
    iou_type = 'GIOU'  # NIOU MPDIOU GIOU, MPDIOU效果更好
    soft_dropout = False
    mask_image = False
