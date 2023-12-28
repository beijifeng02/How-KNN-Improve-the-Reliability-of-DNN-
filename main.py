import torch

from cache.extract_feature import extract_feature
from cfgs.default_cfg import cfg, load_cfg_fom_args
from commons.utils import set_seed, evaluate
from commons.logger import logger
from datasets.utils import build_dataloader
from models.utils import build_model
from algorithms.estimator import build_estimator
from algorithms.calibrator import TemperatureScaling
import faiss


description = "Experiment for <How KNN improves the reliability of DNN>"
load_cfg_fom_args(description)
set_seed(cfg)
model = build_model(cfg)
trainloader, calibloader, testloader = build_dataloader(cfg)

train_features, train_logits, train_labels = extract_feature(model, trainloader, cfg, mode="train")
test_features, test_logits, test_labels = extract_feature(model, testloader, cfg, mode="test")
calib_features, calib_logits, calib_labels = extract_feature(model, calibloader, cfg, mode="calib")
