from loguru import logger
import config
from src.core import YAMLConfig
from utils.trainer import DetTrainer
import torch

seed = 10  # 10 - best_stat: {'epoch': 39, 'coco_eval_bbox': 0.49207939046544713}
torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)      # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子

"""
configs/runtime.yml 中包括了运行中所涉及的超参数，比如scaler amp ema等
configs/rtdetr/rtdetr_r101vd_6x_coco.yml  中包括了模型相关的超参数，比如PResNet的深度、RTDETRTransformer的结构参数以及重写optimizer的参数
    并且还规定了这个网络结构涉及哪些yml文件，在__include__有标明   
configs/rtdetr/include/dataloader.yml  中包括了数据加载的超参数
configs/rtdetr/include/optimizer.yml  包括了更多具体的optimizer和lr_scheduler参数
configs/rtdetr/include/rtdetr_r50vd.ym l 中包括了的具体的网络结构参数，rtdetr_r101vd_6x_coco.yml只修改了部分与r50不同的参数
configs/dataset/coco_detection.yml  包括了数据集的参数，以及batch等

现在要做的工作是首先跑一遍RTDETR r50
再用yolov7跑一遍
再用yolov8跑一遍
之后，用repvgg代替RTDETR的backbone跑一遍，看看多少提点

"""

logger.add('result/rtdetr_r50_MPD/training_log.txt')

if __name__ == '__main__':
    # 初始化
    cfg = YAMLConfig(config.model_yaml,
                     resume=config.resume,
                     use_amp=config.use_amp,
                     tuning=config.tuning)

    # 训练器
    trainer = DetTrainer(cfg)

    logger.warning('start training...')
    # 开始训练或测试
    if config.test_only:
        trainer.val()
    else:
        trainer.training()
