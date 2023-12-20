import argparse

from commons.utils import set_seed, evaluate
from datasets.utils import build_dataloader
from cache.utils import build_classifier
from algorithms.estimator import KNNDistance, GMMAtypicalityEstimator
from algorithms.calibrator import TemperatureScaling


def main():
    parser = argparse.ArgumentParser(description="Experiment for 'How-KNN-Improve-the-Reliability-of-DNN'")
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--model', type=str, default='resnet50', help='model')
    parser.add_argument('--batch_size', type=int, default=256, help='model')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = main()
    set_seed(args.seed)
    model = build_classifier(args.model)
    trainloader, calibloder, testloader = build_dataloader(batch_size=args.batch_size)
    train_feature, train_logits, train_labels = model.run_and_cache_outputs(trainloader)
    _, calib_logits, calib_labels = model.run_and_cache_outputs(calibloder)
    test_feature, test_logits, test_labels = model.run_and_cache_outputs(testloader, train=False)

    # calibration
    calibrator = TemperatureScaling()
    calibrator.fit(calib_logits, calib_labels)
    test_logits = calibrator.calibrate(test_logits, softmax=False)

    # calculate atypicality
    estimator = KNNDistance()
    # estimator = GMMAtypicalityEstimator()
    estimator.fit(train_feature, train_labels)
    atypicality = estimator.compute_atypicality(test_feature)
    data = evaluate(test_labels, test_logits, atypicality)
    print(data)