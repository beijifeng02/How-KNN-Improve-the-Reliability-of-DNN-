# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration file (powered by YACS)."""

import argparse
import os
from datetime import datetime
from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as CfgNode

# Global config object (example usage: from core.config import cfg)
_C = CfgNode()
cfg = _C

# ----------------------------- Data options -------------------------------- #
_C.DATA = CfgNode()

# dataset name
_C.DATA.NAME = 'cifar10'

# class in dataset
_C.DATA.CLASS = 10

# length of dataset
_C.DATA.LENGTH = 10000

# Data directory
_C.DATA.DATA_DIR = "/data/dataset/"

# Data loading worker number
_C.DATA.NUM_WORKERS = 8

# calibration number
_C.DATA.CALIB_NUM = 2000

# ----------------------------- Cache options ------------------------------- #
_C.CACHE = CfgNode()

# cache directory for training
_C.CACHE.TRAIN_DIR = "cache/caches/imagenet/train"

# cache directory for testing
_C.CACHE.TEST_DIR = "cache/caches/imagenet/test"

# cache directory for calibrating
_C.CACHE.CALIB_DIR = "cache/caches/imagenet/calib"

# ----------------------------- Model options ------------------------------- #
_C.MODEL = CfgNode()

# Model name
_C.MODEL.ARCH = 'resnet50'

# pretrained models directory
_C.MODEL.CKPT_DIR = None

# ------------------------------- Testing options --------------------------- #
_C.TEST = CfgNode()

# Batch size for evaluation
_C.TEST.BATCH_SIZE = 128

# save models or not
_C.TEST.SAVE_MODEL = True

# only test, not train
_C.TEST.TEST_ONLY = False

# seed
_C.TEST.SEED = 42

# distance: knn or gmm
_C.TEST.DISTANCE = "knn"

# K
_C.TEST.K = 3

# experiment
_C.TEST.EXPERIMENT = "atypicality"

# --------------------------------- Default config -------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


def update_cfg_from_args(cfg: CfgNode, args: argparse.Namespace, key_mapping: dict):
    for key, value in key_mapping.items():
        if hasattr(args, value):
            keys = key.split('.')
            if len(keys) == 2:  # grandchildren in default nodes
                getattr(cfg, keys[0])[keys[1]] = getattr(args, value)
            elif len(keys) == 1:  # children in new nodes
                setattr(cfg, keys[0], getattr(args, value))


def diff_cfg_nodes(cfg1: CfgNode, cfg2: CfgNode, path=""):
    diff_nodes = []

    # key1
    for key in cfg1.keys():
        new_path = path + "." + key if path else key

        if key not in cfg2:
            diff_nodes.append(new_path)

        elif isinstance(cfg1[key], CfgNode) and isinstance(cfg2[key], CfgNode):
            diff_nodes.extend(diff_cfg_nodes(cfg1[key], cfg2[key], new_path))

    # key2
    for key in cfg2.keys():
        new_path = path + "." + key if path else key
        if key not in cfg1:
            diff_nodes.append(new_path)

    return diff_nodes


def merge_from_file(cfg_file):
    with g_pathmgr.open(cfg_file, "r") as f:
        cfg = _C.load_cfg(f)
        # print(cfg, 'cfg')
    for key in cfg:
        if key not in _C:
            if isinstance(cfg[key], CfgNode):
                _C[key] = CfgNode()
            else:
                _C[key] = cfg[key]
        else:
            _C[key].merge_from_other_cfg(cfg[key])

    _C.merge_from_other_cfg(cfg)


def load_cfg_fom_args(description="Config options."):
    """Load config from command line args and set any specified options."""
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=description)

    # ------------------------------------ configs ----------------------------------------#
    # parser.add_argument("--cfg", dest="cfg_file", type=str, required=True,
    #                     help="Config file location")

    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="See conf.py for all options")

    ############################ add your own hyperparameter here ############################
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--model', type=str, default='resnet50', help='model')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
    parser.add_argument('--batch_size', type=int, default=256, help='test batch size')
    parser.add_argument('--distance', type=str, default="knn", help='distance')

    key_mapping = {
        "TEST.SEED": "seed",
        "MODEL.ARCH": "model",
        "DATA.NAME": "dataset",
        "TEST.BATCH_SIZE": "batch_size",
        "TEST.DISTANCE": "distance"
    }
    ##########################################################################################
    args = parser.parse_args()

    # load ckpt directory
    if args.dataset == "svhn":
        getattr(cfg, "MODEL")["CKPT_DIR"] = f"ckpt/cifar10/{args.model}.pth"
    else:
        getattr(cfg, "MODEL")["CKPT_DIR"] = f"ckpt/{args.dataset}/{args.model}.pth"

    cfg_file = f"cfgs/{args.dataset}.yaml"
    merge_from_file(cfg_file)
    cfg.merge_from_list(args.opts)
    update_cfg_from_args(cfg, args, key_mapping)
