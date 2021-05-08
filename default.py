from yacs.config import CfgNode as CN

_C = CN()

# Train
_C.TRAIN = CN()
_C.TRAIN.OPT = "sgd"  # adam or sgd
_C.TRAIN.WORKERS = 8
_C.TRAIN.LR = 0.01
_C.TRAIN.LR_DECAY_STEP = 20
_C.TRAIN.LR_DECAY_RATE = 0.2
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 0.0
_C.TRAIN.EPOCHS = 80
_C.TRAIN.AGE_STDDEV = 1.0

# Test
_C.TEST = CN()
_C.TEST.WORKERS = 8
