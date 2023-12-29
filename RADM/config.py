# ========================================
# Modified by Fengheng Li
# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_radm_config(cfg):
    """
    Add config for RADM
    """
    cfg.MODEL.RADM = CN()
    cfg.MODEL.RADM.NUM_CLASSES = 4
    cfg.MODEL.RADM.NUM_PROPOSALS = 100
    cfg.MODEL.RADM.withVTRAM = True
    cfg.MODEL.RADM.withGRAM = True
    cfg.MODEL.RADM.NMS_THRESH = 0.15
    cfg.MODEL.RADM.CLASS_THRESH = 0.25
    # Dataset
    cfg.DATASETS.TEXT_FEATURE_PATH = ''
    cfg.DATASETS.DATASET_PATH = ''
    # RCNN Head.
    cfg.MODEL.RADM.NHEADS = 8
    cfg.MODEL.RADM.DROPOUT = 0.0
    cfg.MODEL.RADM.DIM_FEEDFORWARD = 2048
    cfg.MODEL.RADM.ACTIVATION = 'relu'
    cfg.MODEL.RADM.HIDDEN_DIM = 256
    cfg.MODEL.RADM.NUM_CLS = 1
    cfg.MODEL.RADM.NUM_REG = 3
    cfg.MODEL.RADM.NUM_HEADS = 6

    # Dynamic Conv.
    cfg.MODEL.RADM.NUM_DYNAMIC = 2
    cfg.MODEL.RADM.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.RADM.CLASS_WEIGHT = 5.0
    cfg.MODEL.RADM.GIOU_WEIGHT = 1.0
    cfg.MODEL.RADM.L1_WEIGHT = 1.0
    cfg.MODEL.RADM.DEEP_SUPERVISION = True
    cfg.MODEL.RADM.NO_OBJECT_WEIGHT = 0.1

    # Focal Loss.
    cfg.MODEL.RADM.USE_FOCAL = True
    cfg.MODEL.RADM.USE_FED_LOSS = False
    cfg.MODEL.RADM.ALPHA = 0.25
    cfg.MODEL.RADM.GAMMA = 2.0
    cfg.MODEL.RADM.PRIOR_PROB = 0.01

    # Dynamic K
    cfg.MODEL.RADM.OTA_K = 5

    # Diffusion
    cfg.MODEL.RADM.SNR_SCALE = 2.0
    cfg.MODEL.RADM.SAMPLE_STEP = 1

    # Inference
    cfg.MODEL.RADM.USE_NMS = True

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    # TTA.
    cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 640, 700, 900, 1000, 1100, 1200, 1300, 1400, 1800, 800)
    cfg.TEST.AUG.CVPODS_TTA = True
    cfg.TEST.AUG.SCALE_FILTER = True
    cfg.TEST.AUG.SCALE_RANGES = ([96, 10000], [96, 10000], 
                                 [64, 10000], [64, 10000],
                                 [64, 10000], [0, 10000],
                                 [0, 10000], [0, 256],
                                 [0, 256], [0, 192],
                                 [0, 192], [0, 96],
                                 [0, 10000])
