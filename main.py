from cfgs.default_cfg import cfg, load_cfg_fom_args
from commons.utils import set_seed, evaluate
from commons.logger import logger
from datasets.utils import build_dataloader
from cache.utils import build_classifier
from algorithms.estimator import build_estimator
from algorithms.calibrator import TemperatureScaling
import faiss


description = "Experiment for <How KNN improves the reliability of DNN>"
load_cfg_fom_args(description)
set_seed(cfg)
model = build_classifier(cfg)
trainloader, calibloader, testloader = build_dataloader(cfg)
train_feature, train_logits, train_labels = model.run_and_cache_outputs(cfg, trainloader, mode="train")
_, calib_logits, calib_labels = model.run_and_cache_outputs(cfg, calibloader, mode="calib")
test_feature, test_logits, test_labels = model.run_and_cache_outputs(cfg, testloader, mode="test")
logger = logger(cfg)

# calibration
calibrator = TemperatureScaling()
calibrator.fit(calib_logits, calib_labels)
test_logits = calibrator.calibrate(test_logits, softmax_bool=False)

# calculate atypicality
estimator = build_estimator(cfg)
estimator.fit(train_feature, train_labels)
# atypicality = estimator.compute_atypicality(test_feature)
# data = evaluate(test_labels, test_logits, atypicality, N_groups=5)
index = faiss.IndexFlatL2(train_feature.shape[1])
index.add(train_feature)

for K in [50]:
    distances, _ = index.search(test_feature, K)
    atypicality = -distances[:, -1]

data = evaluate(test_labels, test_logits, atypicality, N_groups=5)
logger.update(atypicality, data)
logger.write()
