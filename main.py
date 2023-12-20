from cfgs.defalut_cfg import cfg, load_cfg_fom_args
from commons.utils import set_seed, evaluate
from datasets.utils import build_dataloader
from cache.utils import build_classifier
from algorithms.estimator import build_estimator
from algorithms.calibrator import TemperatureScaling


description = "Experiment for <How KNN improves the reliability of DNN>"
load_cfg_fom_args(description)
set_seed(cfg)
model = build_classifier(cfg)
trainloader, calibloder, testloader = build_dataloader(cfg)
train_feature, train_logits, train_labels = model.run_and_cache_outputs(cfg, trainloader, mode="train")
_, calib_logits, calib_labels = model.run_and_cache_outputs(cfg, calibloder, mode="calib")
test_feature, test_logits, test_labels = model.run_and_cache_outputs(cfg, testloader, mode="test")

# calibration
calibrator = TemperatureScaling()
calibrator.fit(calib_logits, calib_labels)
test_logits = calibrator.calibrate(test_logits, softmax_bool=False)

# calculate atypicality
estimator = build_estimator(cfg)
estimator.fit(train_feature, train_labels)
atypicality = estimator.compute_atypicality(test_feature)
data = evaluate(test_labels, test_logits, atypicality, N_groups=5)
print(data)
