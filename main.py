import argparse

from commons.utils import set_seed
from datasets.utils import build_dataloader
from cache.utils import build_classifier


def main():
    parser = argparse.ArgumentParser(description="Experiment for 'How-KNN-Improve-the-Reliability-of-DNN'")
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--model', type=str, default='resnet50', help='model')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = main()
    set_seed(args.seed)
    model = build_classifier(args.model)
    trainloader, testloader = build_dataloader()
    train_feature, train_logits, train_labels = model.run_and_cache_outputs(trainloader)
    test_feature, test_logits, test_labels = model.run_and_cache_outputs(testloader, train=False)