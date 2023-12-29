import torch
import numpy as np
import pandas as pd

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

# calibration
calibrator = TemperatureScaling()
calibrator.fit(calib_logits, calib_labels)
test_logits = calibrator.calibrate(test_logits, softmax_bool=False)

normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
prepos_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(448, 960)]))

train_features = prepos_feat(train_features)
test_features = prepos_feat(test_features)

index = faiss.IndexFlatL2(train_features.shape[1])
index.add(train_features)
K = 20

D, _ = index.search(test_features, K)
scores_in = D[:,-1]
data = evaluate(test_labels, test_logits, scores_in, N_groups=5)
atypicality = pd.DataFrame(scores_in, columns=["atypicality"])
data.to_csv(f'{cfg.DATA.NAME}_{cfg.MODEL.ARCH}.csv', index=False)
atypicality.to_csv(f"{cfg.DATA.NAME}_{cfg.MODEL.ARCH}_atypicality.csv", index=False)
