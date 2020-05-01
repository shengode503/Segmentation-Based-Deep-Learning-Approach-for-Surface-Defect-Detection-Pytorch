# Import
from yacs.config import CfgNode as CN

_C = CN()
_C.DEVICE = '0'  # 0, 1, 'cpu'
_C.WORKERS = 4
_C.PIN_MEMORY = True
_C.PRINT_FREQ = 30

_C.CHECKPOINT_PATH = 'checkpoints'

' CUDNN '
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# Dataset
_C.DATASET = CN()
_C.DATASET.ROOT = 'datasets/'
_C.DATASET.NAME = 'KolektorSDD/'
_C.DATASET.IMG_RESIZE = [512, 1408]


' TRAIN '
_C.TRAIN = CN()

_C.TRAIN.TRAIN_MODEL = 'DecNet'  # 'SegNet', 'DecNet', 'SegDecNet'

_C.TRAIN.DATA_SHUFFLE = True

# Learning Rate
_C.TRAIN.LR = 1e-3
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.SCHEDULE = [180, 250, 350, 450]

# Optimizer
_C.TRAIN.OPTIMIZER = 'sgd'
_C.TRAIN.MOMENTUM = 0
_C.TRAIN.WEIGHT_DECAY = 0

# Epoch
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 500

# Batch
_C.TRAIN.TRAIN_BATCH = 2  # !!!
_C.TRAIN.snapshot = 10

# Augmentation
_C.TRAIN.IS_FLIP = True
_C.TRAIN.FLIP_PROB = 0.2

_C.TRAIN.IS_DILATE = True
_C.TRAIN.DILATE_PROB = 0.3

_C.TRAIN.IS_ROT_SCAL = True
_C.TRAIN.ROT_SCAL_PROB = 0.8
_C.TRAIN.ROT_RATE = 360
_C.TRAIN.SCAL_RATE = 0.2
_C.TRAIN.SHIFT_RATE = 0.4


def get_cfg_defaults():
  return _C.clone()


if __name__ == '__main__':

    import os

    # # Save
    cfg = get_cfg_defaults()
    cfg.freeze()

    root = 'experiments/'
    with open(os.path.join(root, '{0}_experiment_0.yaml'.format(cfg.TRAIN.TRAIN_MODEL)), 'w') as f:
        print(cfg, file=f)






