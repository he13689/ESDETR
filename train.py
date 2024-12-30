from loguru import logger
import config
from src.core import YAMLConfig
from utils.trainer import DetTrainer
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=FutureWarning)

'''
记录： 
1. configs/rtdetr/rtdetr_r50vd_6x_coco.yml测试了原始网络的性能，结果在 result/rtdetr
2. configs/rtdetr/mdetrv2.yml 测试在加入encover之后的结果 result/rtdetr
3. configs/rtdetr/mdetr.yml 测试修改后的backbone

test1:
softhat = 0.1
mask_percent = 0.1
noise_intense = 0.1

test2:
softhat = 0.1
mask_percent = 0.2
noise_intense = 0.1
'''

logger.add('result/rtdetr101/training_log.txt')

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
